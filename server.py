import align,drdata 
import io

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import SocketServer,cgi
from sys import version as python_version
from cgi import parse_header, parse_multipart
import random
from cgi import FieldStorage


from urlparse import parse_qs
from BaseHTTPServer import BaseHTTPRequestHandler

delim = "newdrtimg"

class S(BaseHTTPRequestHandler):
    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'image/png')
        self.end_headers()

    def do_GET(self):
        self._set_headers()

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        self._set_headers()

        print "received POST to path "+self.path
        print "client address: %s:%s "%(self.client_address)

        self.data_string = self.rfile.read(int(self.headers['Content-Length']))

        ctype, pdict = parse_header(self.headers['content-type'])
        form = cgi.FieldStorage(io.BytesIO(self.data_string),
            headers={
                'content-type':self.headers['content-type'],
                'length':len(self.data_string)
                },
            environ={'REQUEST_METHOD': 'POST'})
        print "form keys:"+str(form.keys())
        print "Data type:"+str(form.getfirst('dataType','(not definded)'))
        if ctype=="multipart/form-data":
            clnm = form.getfirst('className')
            snm = form.getfirst('setName')
            vid = form.getfirst('vidId')
            
            images = form.getlist('image')
            print "Parsed %d images"%(len(images))
            i=  0
            for img in images:
                drdata.save_train_img(clnm,snm,"%s-%d"%(vid,i), img)
                i = i+1
            self.send_response(200)
        else:
            self.send_response(500)
        self.end_headers()
        response = align.hw(self.data_string)
        self.wfile.write(response)

        return

    def parse_POST(self):
        ctype, pdict = parse_header(self.headers['content-type'])
        if ctype == 'multipart/form-data':
            postvars = parse_multipart(self.rfile, pdict)
            print "vars",postvars
        elif ctype == 'application/x-www-form-urlencoded':
            length = int(self.headers['content-length'])
            postvars = parse_qs(
                    self.rfile.read(length), 
                    keep_blank_values=1)
        else:
            print "couldnt parse"
            postvars = {}
        return postvars

def run(server_class=HTTPServer, handler_class=S, port=8899):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print 'Starting httpd...'
    httpd.serve_forever()

if __name__ == "__main__":
    from sys import argv

if len(argv) == 2:
    run(port=int(argv[1]))
else:
    run()
