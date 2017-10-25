import io

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
import SocketServer,cgi
from sys import version as python_version
from cgi import parse_header, parse_multipart
import random
from cgi import FieldStorage

from drmodel import DrModel
import align,drdata 

from urlparse import parse_qs
from BaseHTTPServer import BaseHTTPRequestHandler

delim = "newdrtimg"

drm = DrModel()
drm.load_model("drmodel")

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

	# get srting from request
        self.data_string = self.rfile.read(int(self.headers['Content-Length']))
        ctype, pdict = parse_header(self.headers['content-type'])

	# compose parser
        form = cgi.FieldStorage(io.BytesIO(self.data_string),
            headers={
                'content-type':self.headers['content-type'],
                'length':len(self.data_string)
                },
            environ={'REQUEST_METHOD': 'POST'})
        print "form keys:"+str(form.keys())

	#  get data type
	dtype = str(form.getfirst('dataType','(not definded)'))
        print "Data type:"+dtype

        if ctype=="multipart/form-data":
	    # we decide what to do basing on data type
	    if (dtype == "train"):
	        self.handle_train(form)
		response = align.hw(self.data_string)

	    elif (dtype == "predict"):	
		response=self.handle_predict(form)
	
            self.send_response(200)
        else:
            self.send_response(500)

        self.wfile.write(response)
        return

    def handle_predict(self,form):
        images = form.getlist('image')
	preds = drm.predict(images)
	print "predictions:",preds
	return preds

    def handle_train(self, form):
            clnm = form.getfirst('className')
            snm = form.getfirst('setName')
            vid = form.getfirst('vidId')
            images = form.getlist('image')
            print "Parsed %d images"%(len(images))

            i=  0
            for img in images:
                drdata.save_train_img(clnm,snm,"%s-%d"%(vid,i), img)
                i = i+1
	
def run(server_class=HTTPServer, handler_class=S, port=8899):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print 'Starting httpd on port %d...'%port
    httpd.serve_forever()

if __name__ == "__main__":
    from sys import argv

if len(argv) == 2:
    run(port=int(argv[1]))
else:
    run()
