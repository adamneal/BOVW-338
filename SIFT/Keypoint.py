# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 10:45:23 2021

@author: HMTyl
"""

class Keypoint(object):
    def __init__(self,x,y,size,angle,response,octave,class_id=-1):
        self.x=x
        self.y=y
        self.size=size
        self.angle=angle
        self.response=response
        self.octave=octave
        self.class_id=class_id
        
