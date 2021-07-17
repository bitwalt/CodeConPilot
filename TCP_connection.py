

import socket
import sys

HOST = '127.0.0.1'
PORT = 8888

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('Socket created')

try:
    s.bind((HOST, PORT))
except socket.error as err:
    print('Bind failed. Error Code : ' + str(err))
    sys.exit()

print('Socket bind complete')

s.listen(10)
print('Socket now listening')

conn, addr = s.accept()
print('Connected with ' + addr[0] + ':' + str(addr[1]))

while True:
    data = conn.recv(1024)
    if not data:
        break
    conn.sendall(data)

conn.close()
s.close()