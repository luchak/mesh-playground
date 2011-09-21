#!/usr/bin/env python

import os

def LoadMeshFromFile(filename):
  def NextNonCommentLine(iterable):
    while True:
      line = iterable.next().strip()
      if line != '' and line[0] != '#':
        return line

  def LoadOFF(iterable):
    assert("OFF" == NextNonCommentLine(iterable))
    num_points, num_faces, num_edges = [int(x) for x in NextNonCommentLine(iterable).split()]

    points = []
    for i in range(num_points):
      points.append([float(x) for x in NextNonCommentLine(iterable).split()])

    faces = []
    for i in range(num_faces):
      faces.append([int(x) for x in NextNonCommentLine(iterable).split()[1:4]])

    # don't bother with edges

    return points, faces

  def LoadSMesh(iterable):
    num_points = int(NextNonCommentLine(iterable).split()[0])
    points = []
    for i in range(num_points):
      points.append([float(x) for x in NextNonCommentLine(iterable).split()[1:4]])

    num_faces = int(NextNonCommentLine(iterable).split()[0])
    faces = []
    for i in range(num_faces):
      faces.append([int(x) for x in NextNonCommentLine(iterable).split()[1:4]])

    return points, faces

  def LoadSTL(iterable):
    assert(NextNonCommentLine(iterable).split()[0] == 'solid')
    
    points = []
    faces = []
    point_index = {}

    while True:
      tokens = NextNonCommentLine(iterable).split()
      if tokens[0] == 'endsolid':
        break
      assert(tokens[0] == 'facet')
      assert(NextNonCommentLine(iterable) == 'outer loop')

      face_points = []
      for i in range(3):
        line = NextNonCommentLine(iterable)
        tokens = line.split()
        assert tokens[0] == 'vertex'

        if line not in point_index:
          point_index[line] = len(points)
          points.append([float(x) for x in tokens[1:4]])
        face_points.append(point_index[line])
      faces.append(face_points)

      assert(NextNonCommentLine(iterable) == 'endloop')
      assert(NextNonCommentLine(iterable) == 'endfacet')

    return points, faces

          

  loaders = {'.off': LoadOFF,
             '.smesh': LoadSMesh,
             '.stl': LoadSTL,
             }

  with open(filename, 'r') as mesh_lines:
    ext = os.path.splitext(filename)[1]
    return TriangleMesh(*(loaders[ext](mesh_lines)))

class TriangleMesh(object):
  def __init__(self, points=None, faces=None):
    self.points = points
    self.faces = faces

  def SaveToSMesh(self, filename):
    with open(filename, 'w') as output:
      output.write('%d 3 0 1\n' % len(self.points))
      for i, p in enumerate(self.points):
        output.write('%d %06f %06f %06f 1\n' % (i, p[0], p[1], p[2]))

      output.write('%d 0\n' % len(self.faces))
      for i, f in enumerate(self.faces):
        output.write('3 %d %d %d\n' % tuple(f))
        
      # Not handling holes right now
      output.write('0\n')

  def SaveToOFF(self, filename):
    with open(filename, 'w') as output:
      output.write('OFF\n')
      output.write('%d %d 0\n' % (len(self.points), len(self.faces)))

      for p in self.points:
        output.write('%06f %06f %06f\n' % tuple(p))

      for f in self.faces:
        output.write('3 %d %d %d\n' % tuple(f))

  def Save(self, filename):
    save_functions = {'.off': self.SaveToOFF,
                      '.smesh': self.SaveToSMesh,
                      }
    save_functions[os.path.splitext(filename)[1]](filename)

  def TransformPoints(self, fn):
    for i, p in enumerate(self.points):
      self.points[i] = fn(p)
