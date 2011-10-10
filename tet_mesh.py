#!/usr/bin/env python

import os

import textfile
import triangle_mesh

def LoadFromNodeFile(filename):
  basename = os.path.splitext(filename)[0]

  vertex_line_tokens = textfile.NonCommentLineTokens(basename + '.node')
  num_vertices = int(vertex_line_tokens.next()[0])
  vertices = []
  boundary_vertices = []
  for i in range(num_vertices):
    current_line_tokens = vertex_line_tokens.next()
    vertices.append([float(x) for x in current_line_tokens[1:4]])
    if (current_line_tokens[4] == '1'):
      boundary_vertices.append(i)


  face_line_tokens = textfile.NonCommentLineTokens(basename + '.face')
  num_faces = int(face_line_tokens.next()[0])
  faces = []
  for i in range(num_faces):
    faces.append([int(x) for x in face_line_tokens.next()[1:4]])

  tet_line_tokens = textfile.NonCommentLineTokens(basename + '.ele')
  num_tets = int(tet_line_tokens.next()[0])
  tets = []
  for i in range(num_tets):
    tets.append([int(x) for x in tet_line_tokens.next()[1:5]])

  return TetMesh(vertices, faces, tets, boundary_vertices)


def LoadFromMeshFile(filename):
  line_tokens = textfile.NonCommentLineTokens(filename)

  assert(line_tokens.next()[0] == 'MeshVersionFormatted')
  line = line_tokens.next()
  assert(line[0] == 'Dimension')
  if (len(line) == 1):
    assert(line_tokens.next()[0] == '3')
  else:
    assert(line[1] == '3')

  assert(line_tokens.next()[0] == 'Vertices')
  num_vertices = int(line_tokens.next()[0])
  boundary_vertices = []
  vertices = []
  for i in range(num_vertices):
    current_line_tokens = line_tokens.next()
    vertices.append([float(x) for x in current_line_tokens[0:3]])
    if int(current_line_tokens[3]) > 0:
      boundary_vertices.append(i)

  assert(line_tokens.next()[0] == 'Triangles')
  num_faces = int(line_tokens.next()[0])
  faces = []
  boundary_faces = []
  for i in range(num_faces):
    current_line_tokens = line_tokens.next()
    faces.append([int(x) - 1 for x in current_line_tokens[0:3]])
    if int(current_line_tokens[3]) > 0:
      boundary_faces.append(i)

  assert(line_tokens.next()[0] == 'Tetrahedra')
  num_tets = int(line_tokens.next()[0])
  tets = []
  for i in range(num_tets):
    tets.append([int(x) - 1 for x in line_tokens.next()[0:4]])

  return TetMesh(vertices, faces, tets, boundary_vertices, boundary_faces)


class TetMesh(object):
  def __init__(self, vertices=None, faces=None, tets=None, boundary_vertices=None, boundary_faces=None):
    self.vertices = vertices if vertices else []
    self.faces = faces if faces else []
    self.tets = tets if tets else []
    self.boundary_vertices = set(boundary_vertices) if boundary_vertices else set()
    self.boundary_faces = set(boundary_faces) if boundary_faces else set()
    self.edges = set()
    for tet in self.tets:
      tet = sorted(tet)
      self.edges.add((tet[0], tet[1]))
      self.edges.add((tet[0], tet[2]))
      self.edges.add((tet[0], tet[3]))
      self.edges.add((tet[1], tet[2]))
      self.edges.add((tet[1], tet[3]))
      self.edges.add((tet[2], tet[3]))

  def Copy(self):
    # Using [:] to copy since that should also work for numpy arrays
    return TetMesh(self.vertices[:], self.faces[:], self.tets[:], set(self.boundary_vertices))

  def Save(self, filename):
    assert(filename.endswith('.node'))
    basename = os.path.splitext(filename)[0]

    with open(basename + '.node', 'w') as vertex_file:
      vertex_file.write('%d 3 0 1\n' % len(self.vertices))
      for i, vertex in enumerate(self.vertices):
        vertex_file.write('%d %06f %06f %06f %d\n' %
            (i, vertex[0], vertex[1], vertex[2], 1 if i in self.boundary_vertices else 0))

    with open(basename + '.face', 'w') as face_file:
      face_file.write('%d 0\n' % len(self.faces))
      for i, face in enumerate(self.faces):
        face_file.write('%d %d %d %d\n' % (i, face[0], face[1], face[2]))

    with open(basename + '.ele', 'w') as tet_file:
      tet_file.write('%d 4 0\n' % len(self.tets))
      for i, tet in enumerate(self.tets):
        tet_file.write('%d %d %d %d %d\n' % (i, tet[0], tet[1], tet[2], tet[3]))

  def BoundaryTriangleMesh(self):
    vertices = []
    faces = []
    vertex_map = {}

    for i,v in enumerate(sorted(list(self.boundary_vertices))):
      vertex_map[v] = i
      vertices.append(self.vertices[v])

    if len(self.boundary_faces) > 0:
      for f in self.boundary_faces:
        faces.append([vertex_map[v] for v in self.faces[f]])
    else:
      for f in self.faces:
        if all([v in self.boundary_vertices for v in f]):
          faces.append([vertex_map[v] for v in f])

    return triangle_mesh.TriangleMesh(vertices, faces)

  def FlipTets(self):
    for i in range(len(self.tets)):
      self.tets[i][2], self.tets[i][3] = self.tets[i][3], self.tets[i][2]

  def TransformVertices(self, fn):
    for i, p in enumerate(self.vertices):
      self.vertices[i] = fn(p)

  def TransformBoundaryVertices(self, fn):
    for i, p in enumerate(self.vertices):
      if i in self.boundary_vertices:
        self.vertices[i] = fn(p)

  # This won't stand on its own -- needs to be part of a loop that re-weights edges to unflip bad tets.
  def LaplacianSmooth(self, reference_mesh):
    # placing this down here for now in case people are having numpy/scipy problems
    import numpy
    import scipy.sparse as sparse
    import scipy.sparse.linalg

    num_vertices = len(self.vertices)
    num_boundary_vertices = len(self.boundary_vertices)
    num_non_boundary_vertices = num_vertices - num_boundary_vertices
    L = sparse.lil_matrix((num_vertices, num_vertices))
    C = sparse.lil_matrix((num_boundary_vertices, num_vertices))
    Cbar = sparse.lil_matrix((num_non_boundary_vertices, num_vertices))

    non_boundary_vertices_seen = 0
    boundary_vertices_seen = 0
    for i in xrange(num_vertices):
      if i in self.boundary_vertices:
        C[boundary_vertices_seen, i] = 1.0
        boundary_vertices_seen += 1
      else:
        Cbar[non_boundary_vertices_seen, i] = 1.0
        non_boundary_vertices_seen += 1
    assert (num_boundary_vertices == boundary_vertices_seen)
    assert (num_non_boundary_vertices == non_boundary_vertices_seen)
    C = sparse.kron(C, sparse.eye(3, 3)).tocsr()
    Cbar = sparse.kron(Cbar, sparse.eye(3, 3)).tocsr()

    edge_boundary_vertices = 0
    edge_non_boundary_vertices = 0
    for v0, v1 in self.edges:
      if v0 in self.boundary_vertices:
        edge_boundary_vertices += 1
      else:
        edge_non_boundary_vertices += 1
      if v1 in self.boundary_vertices:
        edge_boundary_vertices += 1
      else:
        edge_non_boundary_vertices += 1

      weight = 1.0
      L[v0,v0] -= weight
      L[v0,v1] += weight

      L[v1,v1] -= weight
      L[v1,v0] += weight
    L = sparse.kron(L, sparse.eye(3, 3)).tocsr()

    xtilde = numpy.array(self.vertices).flatten()
    y = numpy.array(reference_mesh.vertices).flatten()
    CbarLTL = Cbar * (L.T * L)
    b = CbarLTL * (y - C.T * (C * xtilde))

    xbar, info = sparse.linalg.cg(CbarLTL * Cbar.T, b)

    x = Cbar.T * xbar + C.T * (C * xtilde)

    self.vertices = x.reshape(numpy.array(self.vertices).shape)

  def BoundingBoxCorners(self):
    mins = [1e10] * 3
    maxes = [-1e10] * 3
    for v in self.vertices:
      for i in range(3):
        mins[i] = min(mins[i], v[i])
        maxes[i] = max(maxes[i], v[i])
    return mins, maxes
  

