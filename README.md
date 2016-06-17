# Quick Start Guide

'''python
import print3d
'''

Load a model dump

'''python
m = print3d.Model(model = 'z40_sn', directory = '/path/to/dumps/', index = 400)
'''

Create STL file

'''python
m.create_stl(filename = 'output.stl')
'''
