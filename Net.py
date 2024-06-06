class NetAsyncServer(object):
    def __init__(self, *, host, port):
        from collections import defaultdict
        from multiprocessing.connection import Listener

        super().__init__()

        self._listener = Listener(address=(host, port), backlog=50)
        self._clients = []
        self._obj = {}
        self._pending = defaultdict(list)
        self._deferred = set()

    def add_object(self, *, obj_id, obj):
        assert obj_id not in self._obj
        self._obj[obj_id] = obj

    class Deferred(object):
        pass

    def safe_send(self, *, conn, payload):
        try:
            conn.send(payload)
        except EOFError:
            self._clients.remove(conn)
            self._deferred.discard(conn)

    def dispatch(self, *, msg, conn):
        (obj_id, op, args, kwargs) = msg
        try:
            assert conn not in self._deferred, 'clients must block on all responses'
            result = getattr(self._obj[obj_id], op)(*args, **kwargs)
            if isinstance(result, NetAsyncServer.Deferred):
                self._pending[(obj_id, op)].append(conn)
                self._deferred.add(conn)
            else:
                self.safe_send(conn=conn, payload=('result', op, result))
        except Exception as e:
            self.safe_send(conn=conn, payload=('exception', op, 'exception {e} handling msg {msg}'))

    def runonce(self, *, timeout):
        from multiprocessing.connection import wait

        not_deferred_clients = [ c for c in self._clients if c not in self._deferred ]
        ready = wait([ self._listener._listener._socket ] + not_deferred_clients, timeout)

        for conn in ready:
            if conn is self._listener._listener._socket:
                try:
                    newclient = self._listener.accept()
                    self._clients.append(newclient)
                except:
                    pass
            else:
                try:
                    msg = conn.recv()
                except EOFError:
                    self._clients.remove(conn)
                    self._deferred.discard(conn)
                else:
                    if msg == 'shutdown':
                        return False
                    elif msg == 'ping':
                        self.safe_send(conn=conn, payload='pong')
                    else:
                        self.dispatch(msg=msg, conn=conn)
        if not ready:
            for obj_id, obj in self._obj.items():
                for typedresulttuple in obj.tick():
                    _, op, _ = typedresulttuple
                    conn = self._pending[(obj_id, op)].pop(0)
                    if conn in self._clients:
                        self._deferred.discard(conn)
                        self.safe_send(conn=conn, payload=typedresulttuple)

        return True

class NetServer(object):
    def __init__(self, *, host, port):
        from multiprocessing.connection import Listener

        super().__init__()

        self._listener = Listener(address=(host, port), backlog=50)
        self._clients = []
        self._obj = {}

    def safe_send(self, *, conn, payload):
        try:
            conn.send(payload)
        except EOFError:
            self._clients.remove(conn)

    def add_object(self, *, obj_id, obj):
        assert obj_id not in self._obj
        self._obj[obj_id] = obj

    def dispatch(self, *, msg):
        try:
            (obj_id, op, args, kwargs) = msg
            return ('result', op, getattr(self._obj[obj_id], op)(*args, **kwargs))
        except Exception as e:
            return ('exception', op, f'exception {e} handling msg {msg}')

    def runonce(self, *, timeout):
        from multiprocessing.connection import wait

        ready = wait([ self._listener._listener._socket ] + self._clients, timeout)

        for conn in ready:
            if conn is self._listener._listener._socket:
                try:
                    newclient = self._listener.accept()
                    self._clients.append(newclient)
                except:
                    pass
            else:
                try:
                    msg = conn.recv()
                except EOFError:
                    self._clients.remove(conn)
                else:
                    if msg == 'shutdown':
                        return False
                    elif msg == 'ping':
                        self.safe_send(conn=conn, payload='pong')
                    else:
                        result = self.dispatch(msg=msg)
                        self.safe_send(conn=conn, payload=result)

        return True

class NetClient(object):
    def __init__(self, *, host, port, obj_id):
        from multiprocessing.connection import Client

        super().__init__()

        self._client = Client((host, port))
        self._obj_id = obj_id

    def ping(self):
        self._client.send('ping')
        return self._client.recv()

    def shutdown(self):
        self._client.send('shutdown')

    def _rpc(self, *, op, args, kwargs):
        self._client.send((self._obj_id, op, args, kwargs))
        recv_type, recv_op, recv_payload = self._client.recv()
        if recv_type == 'exception':
            raise ValueError((recv_op, recv_payload))
        if recv_op != op:
            raise ValueError(f'obj_id {self._obj_id}: request for {op} but reply for {recv_op}')
        else:
            return recv_payload

    def __getattr__(self, name):
        return lambda *args, **kwargs: self._rpc(op=name, args=args, kwargs=kwargs)

def send_shutdown(*, host, port):
    client = NetClient(host=host, port=port, obj_id=None)
    client.shutdown()

def wait_for_server(*, host, port, alive_check):
    for _ in range(1200):
        try:
            client = NetClient(host=host, port=port, obj_id=None)
            return client.ping() == 'pong'
        except ConnectionRefusedError:
            if not alive_check():
                return False

            from time import sleep
            sleep(1)
