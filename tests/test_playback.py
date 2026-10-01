import asyncio
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import patch

from yarl import URL

from plexio.plex.playback import (
    _keepalive_loop,
    _position_ms,
    _timeline,
    direct_play_url,
    proxy_playback,
    start_keepalive,
)


class FakeContent:
    def __init__(self, chunks, *, block_after=False):
        self.chunks = list(chunks)
        self.block_after = block_after

    async def read(self, _size):
        if self.chunks:
            return self.chunks.pop(0)
        if self.block_after:
            await asyncio.Event().wait()
        return b''


class FakeResponse:
    def __init__(
        self,
        *,
        headers=None,
        status=200,
        chunks=(),
        block_after=False,
    ):
        self.headers = headers or {}
        self.status = status
        self.content = FakeContent(chunks, block_after=block_after)
        self.closed = False
        self.read_called = False

    def __await__(self):
        async def resolve():
            return self

        return resolve().__await__()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        self.close()
        return False

    async def read(self):
        self.read_called = True
        return b''

    def close(self):
        self.closed = True


class FakeClient:
    def __init__(self, stream_response):
        self.stream_response = stream_response
        self.calls = []
        self.timeline_responses = []

    def get(self, url, **kwargs):
        self.calls.append(('GET', url, kwargs))
        if str(url).endswith(':/timeline') or '/:/timeline?' in str(url):
            response = FakeResponse()
            self.timeline_responses.append(response)
            return response
        return self.stream_response

    def head(self, url, **kwargs):
        self.calls.append(('HEAD', url, kwargs))
        return self.stream_response


class KeepaliveResponse(FakeResponse):
    def __init__(self, *, payload=None, status=200):
        super().__init__(status=status)
        self.payload = payload

    async def json(self, content_type=None):
        return self.payload


def sessions_payload(rating_key='42', identifier='plexio-session-id'):
    return {
        'MediaContainer': {
            'Metadata': [
                {'ratingKey': rating_key, 'Client': {'identifier': identifier}},
            ],
        },
    }


class KeepaliveClient:
    """Client that answers /status/sessions from a scripted list.

    Once the script runs out the session is reported gone, so a test that
    forgets to end the loop still terminates instead of hanging.
    """

    def __init__(self, session_payloads=(), *, sessions_status=200):
        self.session_payloads = list(session_payloads)
        self.sessions_status = sessions_status
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append(('GET', url, kwargs))
        if '/status/sessions' not in str(url):
            return FakeResponse()
        if self.sessions_status >= 400:
            return KeepaliveResponse(status=self.sessions_status)
        if self.session_payloads:
            return KeepaliveResponse(payload=self.session_payloads.pop(0))
        return KeepaliveResponse(payload=sessions_payload(identifier='someone-else'))

    def timeline_states(self):
        return [
            url.query['state']
            for _, url, _ in self.calls
            if '/:/timeline' in str(url)
        ]


class PlaybackPositionTests(TestCase):
    def test_position_uses_range_offset_and_elapsed_time(self):
        self.assertEqual(
            _position_ms(
                total=1_000,
                start=500,
                duration_ms=100_000,
                started_at=10,
                now=22.5,
            ),
            62_500,
        )

    def test_position_is_capped_at_duration(self):
        self.assertEqual(
            _position_ms(
                total=1_000,
                start=900,
                duration_ms=100_000,
                started_at=10,
                now=30,
            ),
            100_000,
        )


class PlaybackProxyTests(IsolatedAsyncioTestCase):
    async def test_proxy_stays_on_configured_streaming_url(self):
        upstream = FakeResponse(
            status=206,
            headers={
                'Content-Length': '1000',
                'Content-Range': 'bytes 0-999/1000',
                'Content-Type': 'video/x-matroska',
            },
        )
        client = FakeClient(upstream)
        request = SimpleNamespace(method='HEAD', headers={})
        configuration = SimpleNamespace(
            streaming_url=URL('http://192.168.50.194:32400'),
            direct_play_connections=[
                (URL('http://192.168.50.194:32400'), None),
                (URL('https://172-20-5-22.example.plex.direct:32400'), None),
            ],
            discovery_url=URL('https://discovery.example.test'),
            access_token='secret',
        )

        response = await proxy_playback(
            request,
            client=client,
            configuration=configuration,
            rating_key='42',
            duration_ms=60_000,
            part_key='/library/parts/1/file.mkv',
            identifier='session-id',
        )

        self.assertEqual(response.status_code, 206)
        self.assertTrue(upstream.closed)
        self.assertEqual([call[1].host for call in client.calls], ['192.168.50.194'])

    async def test_timeline_sends_query_and_releases_response(self):
        client = FakeClient(FakeResponse())

        sent = await _timeline(
            client,
            url=URL('https://plex.example.test'),
            token='secret',
            rating_key='42',
            state='playing',
            time_ms=12_000,
            duration_ms=60_000,
            identifier='session-id',
        )

        self.assertTrue(sent)
        _, url, kwargs = client.calls[0]
        self.assertEqual(url.path, '/:/timeline')
        self.assertEqual(url.query['ratingKey'], '42')
        self.assertEqual(url.query['time'], '12000')
        self.assertEqual(url.query['X-Plex-Token'], 'secret')
        self.assertEqual(
            kwargs['headers']['X-Plex-Client-Identifier'],
            'plexio-session-id',
        )
        self.assertTrue(client.timeline_responses[0].read_called)
        self.assertTrue(client.timeline_responses[0].closed)

    @patch('plexio.plex.playback.monotonic')
    async def test_proxy_reports_elapsed_progress_and_forwards_range(
        self,
        monotonic,
    ):
        monotonic.side_effect = [100.0, 112.0]
        upstream = FakeResponse(
            headers={
                'Content-Length': '1000',
                'Accept-Ranges': 'bytes',
                'Content-Type': 'video/x-matroska',
            },
            chunks=(b'first', b'second'),
        )
        client = FakeClient(upstream)
        configuration = SimpleNamespace(
            streaming_url=URL('https://plex.example.test'),
            discovery_url=URL('https://plex.example.test'),
            access_token='secret',
        )
        request = SimpleNamespace(
            method='GET',
            headers={'range': 'bytes=0-'},
        )

        response = await proxy_playback(
            request,
            client=client,
            configuration=configuration,
            rating_key='42',
            duration_ms=60_000,
            part_key='/library/parts/1/file.mkv',
            identifier='session-id',
        )
        body = b''.join([chunk async for chunk in response.body_iterator])

        self.assertEqual(body, b'firstsecond')
        self.assertTrue(upstream.closed)
        self.assertEqual(client.calls[0][2]['headers']['Range'], 'bytes=0-')
        self.assertEqual(
            client.calls[0][2]['headers']['X-Plex-Client-Identifier'],
            'plexio-session-id',
        )
        self.assertIsNone(client.calls[0][2]['timeout'].sock_read)
        timeline_urls = [url for _, url, _ in client.calls if '/:/timeline' in str(url)]
        self.assertEqual(
            [url.query['state'] for url in timeline_urls],
            ['playing', 'stopped'],
        )
        self.assertEqual(
            [url.query['time'] for url in timeline_urls],
            ['0', '12000'],
        )

    @patch('plexio.plex.playback.UPSTREAM_READ_TIMEOUT', 0.01)
    @patch('plexio.plex.playback.PING_INTERVAL', 0.01)
    async def test_heartbeat_continues_while_downstream_is_idle(self):
        upstream = FakeResponse(
            headers={
                'Content-Length': '1000',
                'Content-Type': 'video/x-matroska',
            },
            chunks=(b'first', b'second'),
        )
        client = FakeClient(upstream)
        configuration = SimpleNamespace(
            streaming_url=URL('https://plex.example.test'),
            discovery_url=URL('https://plex.example.test'),
            access_token='secret',
        )
        request = SimpleNamespace(method='GET', headers={})

        response = await proxy_playback(
            request,
            client=client,
            configuration=configuration,
            rating_key='42',
            duration_ms=60_000,
            part_key='/library/parts/1/file.mkv',
            identifier='session-id',
        )
        iterator = response.body_iterator

        self.assertEqual(await anext(iterator), b'first')
        await asyncio.sleep(0.045)

        states = [
            url.query['state']
            for _, url, _ in client.calls
            if '/:/timeline' in str(url)
        ]
        self.assertGreaterEqual(states.count('playing'), 2)
        self.assertEqual(await anext(iterator), b'second')

        await iterator.aclose()
        states = [
            url.query['state']
            for _, url, _ in client.calls
            if '/:/timeline' in str(url)
        ]
        self.assertEqual(states[-1], 'stopped')
        self.assertTrue(upstream.closed)

    @patch('plexio.plex.playback.UPSTREAM_READ_TIMEOUT', 0.01)
    async def test_active_upstream_stall_times_out_and_closes_response(self):
        upstream = FakeResponse(
            headers={
                'Content-Length': '1000',
                'Content-Type': 'video/x-matroska',
            },
            chunks=(b'first',),
            block_after=True,
        )
        client = FakeClient(upstream)
        configuration = SimpleNamespace(
            streaming_url=URL('https://plex.example.test'),
            discovery_url=URL('https://plex.example.test'),
            access_token='secret',
        )
        request = SimpleNamespace(method='GET', headers={})

        response = await proxy_playback(
            request,
            client=client,
            configuration=configuration,
            rating_key='42',
            duration_ms=60_000,
            part_key='/library/parts/1/file.mkv',
            identifier='session-id',
        )
        iterator = response.body_iterator

        self.assertEqual(await anext(iterator), b'first')
        with self.assertRaises(TimeoutError):
            await anext(iterator)

        self.assertTrue(upstream.closed)
        states = [
            url.query['state']
            for _, url, _ in client.calls
            if '/:/timeline' in str(url)
        ]
        self.assertEqual(states, ['playing', 'stopped'])

    async def test_head_probe_returns_headers_without_streaming_or_timeline(self):
        upstream = FakeResponse(
            status=206,
            headers={
                'Content-Length': '1000',
                'Content-Range': 'bytes 0-999/1000',
                'Content-Type': 'video/x-matroska',
            },
        )
        client = FakeClient(upstream)
        request = SimpleNamespace(method='HEAD', headers={})
        configuration = SimpleNamespace(
            streaming_url=URL('https://plex.example.test'),
            discovery_url=URL('https://plex.example.test'),
            access_token='secret',
        )

        response = await proxy_playback(
            request,
            client=client,
            configuration=configuration,
            rating_key='42',
            duration_ms=60_000,
            part_key='/library/parts/1/file.mkv',
            identifier='session-id',
        )

        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.headers['content-range'], 'bytes 0-999/1000')
        self.assertTrue(upstream.closed)
        self.assertEqual([method for method, _, _ in client.calls], ['HEAD'])


class PlaybackKeepaliveTests(IsolatedAsyncioTestCase):
    def _configuration(self):
        return SimpleNamespace(
            streaming_url=URL('http://192.168.50.194:32400'),
            discovery_url=URL('https://plex.example.test'),
            access_token='secret',
        )

    def _run_loop(self, client, duration_ms):
        return _keepalive_loop(
            client,
            url=URL('https://plex.example.test'),
            token='secret',
            rating_key='42',
            duration_ms=duration_ms,
            identifier='session-id',
        )

    def test_direct_play_url_carries_this_install_as_the_client(self):
        parsed = URL(
            direct_play_url(
                configuration=self._configuration(),
                part_key='/library/parts/1/file.mkv',
                identifier='session-id',
            )
        )

        self.assertEqual(parsed.path, '/library/parts/1/file.mkv')
        self.assertEqual(parsed.query['X-Plex-Token'], 'secret')
        self.assertEqual(
            parsed.query['X-Plex-Client-Identifier'],
            'plexio-session-id',
        )
        self.assertEqual(parsed.query['X-Plex-Product'], 'Plexio')

    async def test_keepalive_redirects_player_to_plex_without_proxying_media(self):
        client = KeepaliveClient()
        with (
            patch('plexio.plex.playback.PING_INTERVAL', 0.01),
            patch('plexio.plex.playback.KEEPALIVE_GRACE', 0),
        ):
            response = start_keepalive(
                client=client,
                configuration=self._configuration(),
                rating_key='42',
                duration_ms=0,
                part_key='/library/parts/1/file.mkv',
                identifier='session-id',
            )
            await asyncio.sleep(0.05)

        self.assertEqual(response.status_code, 302)
        self.assertEqual(
            URL(response.headers['location']).query['X-Plex-Client-Identifier'],
            'plexio-session-id',
        )
        self.assertEqual(
            [url for _, url, _ in client.calls if '/library/parts/' in str(url)],
            [],
        )
        self.assertEqual(client.timeline_states(), ['playing', 'stopped'])

    @patch('plexio.plex.playback.PING_INTERVAL', 0.01)
    async def test_keepalive_reports_playing_until_plex_drops_the_session(self):
        client = KeepaliveClient(
            [sessions_payload(), sessions_payload(), sessions_payload()]
        )

        await self._run_loop(client, 60_000)

        self.assertEqual(
            client.timeline_states(),
            ['playing', 'playing', 'playing', 'playing', 'stopped'],
        )

    @patch('plexio.plex.playback.PING_INTERVAL', 0.01)
    @patch('plexio.plex.playback.KEEPALIVE_GRACE', 0)
    async def test_keepalive_keeps_reporting_when_sessions_query_fails(self):
        client = KeepaliveClient(sessions_status=503)

        await self._run_loop(client, 50)

        session_queries = [
            url for _, url, _ in client.calls if '/status/sessions' in str(url)
        ]
        self.assertGreaterEqual(len(session_queries), 2)
        states = client.timeline_states()
        self.assertGreaterEqual(states.count('playing'), 2)
        self.assertEqual(states[-1], 'stopped')

    @patch('plexio.plex.playback.PING_INTERVAL', 0.01)
    @patch('plexio.plex.playback.KEEPALIVE_GRACE', 0)
    async def test_keepalive_stops_at_the_runtime_cap(self):
        client = KeepaliveClient([sessions_payload()] * 50)

        await self._run_loop(client, 30)

        states = client.timeline_states()
        self.assertGreaterEqual(states.count('playing'), 2)
        self.assertEqual(states[-1], 'stopped')
