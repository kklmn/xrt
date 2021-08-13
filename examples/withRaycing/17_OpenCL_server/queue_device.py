# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 23:58:40 2021

@author: roman
"""

import zmq

def main():

    try:
        context = zmq.Context(1)
        # Socket facing clients
        frontend = context.socket(zmq.XREP)
        frontend.bind("tcp://*:15559")
        # Socket facing services
        backend = context.socket(zmq.XREQ)
        backend.bind("tcp://*:15560")

        zmq.device(zmq.QUEUE, frontend, backend)
    except:
        print("bringing down zmq device")
    finally:
        pass
        frontend.close()
        backend.close()
        context.term()

if __name__ == "__main__":
    main()