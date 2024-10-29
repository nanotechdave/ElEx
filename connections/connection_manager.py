# connection_manager.py
class ConnectionManager:
    def __init__(self, address, protocol="GPIB"):
        # Initialize connection based on protocol
        self.address = address
        self.protocol = protocol
        # Code to initialize the connection

    def send(self, command):
        # Code to send command
        pass

    def receive(self):
        # Code to receive data
        pass
