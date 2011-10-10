#!/usr/bin/env python

import os
import random

import numpy

def LoadMeshFromFile(filename):
  def NextNonCommentLine(iterable):
    while True:
      line = iterable.next().strip()
      if line != '' and line[0] != '#':
        return line

  def LoadOFF(iterable):
    first_line = NextNonCommentLine(iterable)
    assert("OFF" == first_line or "COFF" == first_line)
    num_vertices, num_faces, num_edges = [int(x) for x in NextNonCommentLine(iterable).split()]

    vertices = []
    for i in range(num_vertices):
      vertices.append([float(x) for x in NextNonCommentLine(iterable).split()])

    faces = []
    for i in range(num_faces):
      faces.append([int(x) for x in NextNonCommentLine(iterable).split()[1:4]])

    # don't bother with edges

    return vertices, faces

  def LoadMesh(iterable):
    assert(NextNonCommentLine(iterable).startswith('MeshVersionFormatted'))
    assert(NextNonCommentLine(iterable) == 'Dimension')
    assert(NextNonCommentLine(iterable) == '3')

    assert(NextNonCommentLine(iterable) == 'Vertices')
    num_vertices = int(NextNonCommentLine(iterable))
    vertices = []
    for i in range(num_vertices):
      vertices.append([float(x) for x in NextNonCommentLine(iterable).split()[0:3]])

    assert(NextNonCommentLine(iterable) == 'Triangles')
    num_faces = int(NextNonCommentLine(iterable))
    faces = []
    for i in range(num_faces):
      faces.append([int(x) - 1 for x in NextNonCommentLine(iterable).split()[0:3]])

    return vertices, faces

  def LoadSMesh(iterable):
    num_vertices = int(NextNonCommentLine(iterable).split()[0])
    vertices = []
    for i in range(num_vertices):
      vertices.append([float(x) for x in NextNonCommentLine(iterable).split()[1:4]])

    num_faces = int(NextNonCommentLine(iterable).split()[0])
    faces = []
    for i in range(num_faces):
      faces.append([int(x) for x in NextNonCommentLine(iterable).split()[1:4]])

    return vertices, faces

  def LoadSTL(iterable):
    assert(NextNonCommentLine(iterable).split()[0] == 'solid')
    
    vertices = []
    faces = []
    vertex_index = {}

    while True:
      tokens = NextNonCommentLine(iterable).split()
      if tokens[0] == 'endsolid':
        break
      assert(tokens[0] == 'facet')
      assert(NextNonCommentLine(iterable) == 'outer loop')

      face_vertices = []
      for i in range(3):
        line = NextNonCommentLine(iterable)
        tokens = line.split()
        assert tokens[0] == 'vertex'

        if line not in vertex_index:
          vertex_index[line] = len(vertices)
          vertices.append([float(x) for x in tokens[1:4]])
        face_vertices.append(vertex_index[line])
      faces.append(face_vertices)

      assert(NextNonCommentLine(iterable) == 'endloop')
      assert(NextNonCommentLine(iterable) == 'endfacet')

    return vertices, faces

  loaders = {'.off': LoadOFF,
             '.mesh': LoadMesh,
             '.smesh': LoadSMesh,
             '.stl': LoadSTL,
             }

  with open(filename, 'r') as mesh_lines:
    ext = os.path.splitext(filename)[1]
    return TriangleMesh(*(loaders[ext](mesh_lines)))

class TriangleMesh(object):
  def __init__(self, vertices=None, faces=None, holes=None):
    self.vertices = vertices if vertices else []
    self.faces = faces if faces else []
    self.holes = holes if holes else []

  def SaveToSMesh(self, filename):
    with open(filename, 'w') as output:
      output.write('%d 3 0 1\n' % len(self.vertices))
      for i, p in enumerate(self.vertices):
        output.write('%d %06f %06f %06f 1\n' % (i, p[0], p[1], p[2]))

      output.write('%d 0\n' % len(self.faces))
      for i, f in enumerate(self.faces):
        output.write('3 %d %d %d\n' % tuple(f))
        
      output.write('%d\n' % len(self.holes))
      for i, h in enumerate(self.holes):
        output.write('%d %06f %06f %06f\n' % (i, h[0], h[1], h[2]))

  def SaveToOFF(self, filename):
    with open(filename, 'w') as output:
      output.write('OFF\n')
      output.write('%d %d 0\n' % (len(self.vertices), len(self.faces)))

      for p in self.vertices:
        output.write('%06f %06f %06f\n' % tuple(p))

      for f in self.faces:
        output.write('3 %d %d %d\n' % tuple(f))

  def Save(self, filename):
    save_functions = {'.off': self.SaveToOFF,
                      '.smesh': self.SaveToSMesh,
                      }
    save_functions[os.path.splitext(filename)[1]](filename)

  def AddBox(self, min_corner, max_corner):
    start_vertex_index = len(self.vertices)

    # Generate all corner vertices
    for i in range(8):
       self.vertices.append(
           [(max_corner[axis] if (i & (1 << axis)) else min_corner[axis])
             for axis in range(3)])

    def AddBoxFace(offsets):
      self.faces.append([start_vertex_index + offset for offset in offsets])
    # Generate faces (normals point outwards)
    AddBoxFace([0, 4, 2])
    AddBoxFace([2, 4, 6])
    AddBoxFace([1, 3, 5])
    AddBoxFace([3, 7, 5])

    AddBoxFace([0, 1, 4])
    AddBoxFace([1, 5, 4])
    AddBoxFace([2, 6, 3])
    AddBoxFace([3, 6, 7])

    AddBoxFace([0, 2, 1])
    AddBoxFace([4, 5, 6])
    AddBoxFace([1, 2, 3])
    AddBoxFace([5, 7, 6])

  def BoundingBoxCorners(self):
    mins = [1e10] * 3
    maxes = [-1e10] * 3
    for v in self.vertices:
      for i in range(3):
        mins[i] = min(mins[i], v[i])
        maxes[i] = max(maxes[i], v[i])
    return mins, maxes

  def TransformVertices(self, fn):
    for i, p in enumerate(self.vertices):
      self.vertices[i] = fn(p)

  def UniformScale(self, factor):
    self.TransformVertices(lambda x: [component * factor for component in x])

  def Translate(self, vector):
    self.TransformVertices(lambda x: [x[i] + vector[i] for i in range(len(x))])

  def SampleNewVerticesOnSurface(self, additional_vertices, noise=0.0):
    vertices = numpy.array(self.vertices)

    def FaceArea(face):
      a = vertices[face[2]] - vertices[face[0]]
      b = vertices[face[1]] - vertices[face[0]]
      result = 0.5 * numpy.linalg.norm(numpy.cross(a, b))
      return result

    face_areas = [FaceArea(face) for face in self.faces]
    print min(face_areas)
    print max(face_areas)
    total_area = sum(face_areas)
    vertex_rate = additional_vertices / total_area

    new_vertices = []
    for i, face in enumerate(self.faces):
      num_new_vertices = int(vertex_rate * face_areas[i])
      a = vertices[face[2]] - vertices[face[0]]
      b = vertices[face[1]] - vertices[face[0]]
      for j in range(num_new_vertices):
        ar = random.random()
        br = random.random()
        if ar + br >= 1.0:
          ar = 1.0 - ar
          br = 1.0 - br
        if noise > 0.0:
          new_vertices.append(ar*a + br*b + vertices[face[0]] + numpy.random.normal(0.0, noise, (3,)))
        else:
          new_vertices.append(ar*a + br*b + vertices[face[0]])

    self.vertices = list(self.vertices)
    self.vertices.extend(new_vertices)


