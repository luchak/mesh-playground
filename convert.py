#!/usr/bin/env python

import sys

import triangle_mesh

mesh = triangle_mesh.LoadMeshFromFile(sys.argv[1])
mesh.Save(sys.argv[2])
