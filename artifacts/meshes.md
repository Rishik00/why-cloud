# The Complete Study Guide to Mesh Generation

**From Zero to Implementation — A Practitioner's Manual**

---

## How to Use This Document

This document is structured as a progressive curriculum. Each chapter builds on the previous one. If you read it front-to-back and implement the exercises suggested at the end of each chapter, you will go from knowing nothing about meshes to being able to implement mesh generation algorithms from scratch.

Chapters 1–3 are foundational: what meshes are, the math beneath them, and how they are stored in memory and on disk. Chapters 4–7 are the algorithmic core: the major families of mesh generation algorithms, explained with enough detail to implement. Chapter 8 covers how to measure whether a mesh is good. Chapter 9 covers subdivision and refinement — how to take a coarse mesh and make it smoother. Chapter 10 covers the modern frontier: neural and learned approaches to mesh generation, which should feel natural given your ML background. Chapter 11 is a curated, opinionated reading list.

---

## Chapter 1: What Is a Mesh, and Why Should You Care?

### 1.1 The Core Idea

A mesh is a discrete approximation of a continuous shape. That is the single most important sentence in this entire document, and everything else follows from it.

Consider a sphere. A mathematical sphere is a continuous surface — every point on it is defined by the equation x² + y² + z² = r². But computers cannot store or manipulate continuous surfaces directly. They need a finite collection of numbers. A mesh solves this by replacing the smooth sphere with a collection of flat polygonal faces (usually triangles) that, when stitched together, approximate the sphere's shape. The more faces you use, the closer the approximation.

This is not merely a computer graphics trick. Meshes are the foundational data structure for:

- **Rendering**: Every 3D object you see in a video game, film, or AR app is a mesh. The GPU takes vertices and triangles and projects them onto your screen.
- **Simulation**: Finite element analysis (FEA) solves partial differential equations over complex domains by discretizing them into meshes. This is how engineers simulate stress in bridges, airflow over wings, and heat transfer in engines.
- **Scientific computing**: Weather models, ocean simulations, astrophysics — all discretize their computational domains into meshes.
- **3D printing**: The STL file your printer reads is literally a triangle mesh.
- **Machine learning on geometry**: Point clouds are unstructured, but meshes give you connectivity — topology — which is exploitable by graph neural networks, mesh convolution operators, and geometric deep learning.

### 1.2 Anatomy of a Mesh

A mesh has three fundamental components:

**Vertices** are points in space. Each vertex is a coordinate tuple — (x, y) in 2D, (x, y, z) in 3D. Vertices are the atoms of a mesh. A mesh with n vertices stores n coordinate tuples.

**Edges** are connections between pairs of vertices. An edge connects vertex i to vertex j, forming a line segment. Edges define the skeleton of the mesh.

**Faces** are closed polygons bounded by edges. A face is defined by an ordered list of vertex indices. In a triangle mesh, every face is a triplet of vertex indices. In a quad mesh, every face is a quadruplet. Faces are what you actually "see" when you render a mesh — they are the surfaces.

The combination of these three elements gives you the mesh's **geometry** (where things are in space, determined by vertex positions) and its **topology** (how things are connected, determined by which vertices form edges and faces). This geometry/topology distinction is critical. Two meshes can have identical topology — the same connectivity graph — but completely different geometry, and vice versa.

### 1.3 The Euler Characteristic and Mesh Topology

For any closed, connected mesh that is topologically equivalent to a sphere (no holes, no handles), a beautiful invariant holds:

```
V - E + F = 2
```

where V is the number of vertices, E the number of edges, and F the number of faces. This is the **Euler formula** for polyhedra. For a surface of genus g (a surface with g handles — a torus has genus 1, a double-torus has genus 2), it generalizes to:

```
V - E + F = 2 - 2g
```

This is not just a curiosity. It is a fundamental constraint on mesh construction. If you are building a mesh and your vertex, edge, and face counts violate this relationship, something is topologically wrong — you have a non-manifold edge, a missing face, or a disconnected component you did not intend.

### 1.4 Manifold vs. Non-Manifold Meshes

A mesh is **manifold** if every point on its surface has a neighborhood that looks like a flat disk (or a half-disk, if the point is on a boundary). Practically, this means:

- Every edge is shared by exactly two faces (interior edge) or exactly one face (boundary edge).
- The faces around every vertex form a single connected fan.

Non-manifold meshes have edges shared by three or more faces, or vertices where the fan of faces splits into disconnected groups. Most mesh processing algorithms assume manifold input. When you load a mesh and something breaks, check for non-manifold elements first — they are the number one source of bugs.

### 1.5 Types of Mesh Elements

In 2D:
- **Triangles**: The workhorse. Any polygon can be decomposed into triangles. Triangles are always planar and always convex, which simplifies almost every algorithm.
- **Quadrilaterals (quads)**: Preferred in certain simulation contexts because they can align with flow directions and produce more uniform element sizes. Quads are not necessarily planar in 3D.

In 3D (volume meshes):
- **Tetrahedra**: The 3D analog of triangles. Four vertices, four triangular faces. Any volume can be decomposed into tetrahedra.
- **Hexahedra**: The 3D analog of quads — six-faced brick-like elements. Harder to generate but preferred for many FEA applications because they converge faster with fewer elements.
- **Prisms and pyramids**: Transitional elements used to connect tetrahedral and hexahedral regions.

**Key insight for implementation**: Start with triangle meshes. They are the simplest, the most universally supported, and every algorithm in this document works on them (or has a triangle-mesh variant). Once you are comfortable with triangle meshes, extending to quads or tetrahedra is conceptually straightforward.

*Exercises*: (1) Take a cube. Count V, E, F. Verify the Euler formula. (2) Take a torus (e.g., from an OBJ file online). Count V, E, F. Verify the genus-1 formula. (3) Write code to load an OBJ file and check if the mesh is manifold by verifying the edge-sharing constraint.

---

## Chapter 2: Mathematical Foundations

This chapter covers the minimum mathematics you need. You do not need to read a textbook on differential geometry, but you do need comfort with these concepts.

### 2.1 Vectors and the Cross Product

You are in 3D space. Vertices are vectors in ℝ³. Two operations are essential:

**Dot product**: a · b = aₓbₓ + aᵧbᵧ + a_zbz. Gives you the cosine of the angle between vectors (when normalized). Used everywhere: lighting, projection, angle computation.

**Cross product**: a × b = (aᵧb_z - a_zbᵧ, a_zbₓ - aₓb_z, aₓbᵧ - aᵧbₓ). Gives you a vector perpendicular to both a and b, with magnitude equal to the area of the parallelogram they span. Used to compute face normals, signed areas, and orientation tests.

### 2.2 Face Normals and Vertex Normals

The **face normal** of a triangle with vertices p₁, p₂, p₃ is:

```
n = (p₂ - p₁) × (p₃ - p₁)
```

Normalized (divided by its magnitude), this gives you the unit normal — the direction the triangle "faces." The winding order matters: if you list vertices counter-clockwise when viewed from the front, the normal points toward the viewer (right-hand rule).

**Vertex normals** are computed by averaging the face normals of all faces incident to a vertex. This averaging is what makes flat-shaded meshes appear smooth (Gouraud/Phong shading). The most common weighting schemes are area-weighted (weight each face normal by the triangle's area) and angle-weighted (weight by the angle at that vertex within each triangle). Angle-weighting generally produces better results.

### 2.3 Barycentric Coordinates

Given a triangle with vertices A, B, C, any point P inside the triangle can be written as:

```
P = λ₁A + λ₂B + λ₃C,   where λ₁ + λ₂ + λ₃ = 1,   λᵢ ≥ 0
```

The λᵢ are the **barycentric coordinates** of P. They tell you how much each vertex contributes to the point's position. They are used for interpolation (of colors, normals, texture coordinates across a triangle's face) and for point-in-triangle tests (P is inside the triangle if and only if all three barycentric coordinates are non-negative).

Computing them is straightforward: the barycentric coordinate λ₁ with respect to vertex A equals the area of triangle PBC divided by the area of triangle ABC, and analogously for the others.

### 2.4 Circumcircles and the Delaunay Criterion

Given a triangle with vertices A, B, C, the **circumcircle** (circumscribed circle) is the unique circle passing through all three vertices. Its center (the circumcenter) is equidistant from all three vertices.

The **Delaunay criterion** states: a triangulation is a Delaunay triangulation if and only if the circumcircle of every triangle contains no other vertex from the point set in its interior. This criterion is central to mesh generation because Delaunay triangulations maximize the minimum angle across all triangles — they avoid thin, badly-shaped triangles as much as the point distribution allows.

The in-circle test (does point D lie inside the circumcircle of triangle ABC?) reduces to evaluating the sign of a 4×4 determinant:

```
| ax-dx  ay-dy  (ax-dx)²+(ay-dy)² |
| bx-dx  by-dy  (bx-dx)²+(by-dy)² | > 0  ⟹  D is inside
| cx-dx  cy-dy  (cx-dx)²+(cy-dy)² |
```

This is the mathematical heart of Delaunay algorithms. If you implement one thing from this chapter, implement this test robustly.

### 2.5 Voronoi Diagrams: The Dual of Delaunay

Given a set of points (called sites), the **Voronoi diagram** partitions the plane into cells, one per site, where each cell contains all points closer to its site than to any other site. The Voronoi diagram is the geometric dual of the Delaunay triangulation: connecting sites whose Voronoi cells share an edge gives you exactly the Delaunay triangulation.

This duality is not just elegant — it is computationally useful. Algorithms often construct one and derive the other. Voronoi diagrams themselves are used in mesh generation (centroidal Voronoi tessellations), spatial queries, and nearest-neighbor searches.

### 2.6 Signed Area and Orientation Tests

The signed area of a triangle with vertices (x₁, y₁), (x₂, y₂), (x₃, y₃) is:

```
A = ½ [(x₂ - x₁)(y₃ - y₁) - (x₃ - x₁)(y₂ - y₁)]
```

If A > 0, the vertices are in counter-clockwise order. If A < 0, clockwise. If A = 0, the points are collinear. This orientation test is the most fundamental geometric predicate. It is used in convex hull construction, point-in-polygon tests, triangulation algorithms, and everywhere else.

**Warning about floating-point arithmetic**: The orientation test and in-circle test are highly sensitive to floating-point errors. When three points are nearly collinear, or four points are nearly co-circular, floating-point evaluation can give the wrong sign, leading to incorrect topology. Jonathan Shewchuk's paper "Robust Adaptive Floating-Point Geometric Predicates" (1997) provides exact arithmetic predicates that handle this. His code is freely available and widely used. If you are implementing Delaunay triangulation for anything beyond toy examples, use exact predicates.

*Exercises*: (1) Implement the cross product, face normal, and barycentric coordinate computation. (2) Implement the circumcircle computation: given three points, find the circumcenter and circumradius. (3) Implement the in-circle test using the determinant formula. Test it on degenerate cases.

---

## Chapter 3: Data Structures for Meshes

How you store a mesh in memory determines what operations are fast and what operations are slow. This chapter covers the major representations, from simplest to most powerful.

### 3.1 The Face List (Indexed Face Set / "Polygon Soup")

The simplest representation. You store two arrays:

```
vertices: [(x₁,y₁,z₁), (x₂,y₂,z₂), ..., (xₙ,yₙ,zₙ)]
faces:    [(i₁,j₁,k₁), (i₂,j₂,k₂), ..., (iₘ,jₘ,kₘ)]
```

Each face is a triplet of indices into the vertex array. This is what OBJ files store. This is what GPUs consume. For rendering, this is sufficient.

**Strengths**: Compact, simple, directly maps to GPU buffers, easy to serialize.

**Weaknesses**: Answering topological queries is expensive. "What faces are adjacent to this face?" requires scanning the entire face list. "What edges exist?" requires building them from face data. There is no explicit edge representation.

This representation is sometimes called a "polygon soup" because without additional processing, you have a pile of disconnected polygons. You can verify they form a proper mesh, but the data structure itself does not enforce or expose connectivity.

### 3.2 The Adjacency List

An improvement: for each vertex, store the list of faces it belongs to. For each face, store the list of adjacent faces (faces sharing an edge). This gives you O(1) access to neighbors, at the cost of more memory and the need to maintain these lists when the mesh changes.

### 3.3 The Half-Edge Data Structure

This is the gold standard for mesh processing. It is elegant, efficient, and makes local mesh traversal trivial. If you implement one data structure from this document, implement this one.

The central idea: every undirected edge in the mesh is split into two **half-edges**, pointing in opposite directions. Each half-edge stores:

```
struct HalfEdge {
    vertex:   Vertex      // the vertex this half-edge points TO
    face:     Face         // the face to the LEFT of this half-edge
    next:     HalfEdge     // the next half-edge around the same face
    prev:     HalfEdge     // the previous half-edge around the same face (optional, derivable from next)
    twin:     HalfEdge     // the opposite half-edge (same geometric edge, opposite direction)
}

struct Vertex {
    position: (x, y, z)
    halfedge: HalfEdge     // ANY outgoing half-edge from this vertex
}

struct Face {
    halfedge: HalfEdge     // ANY half-edge bounding this face
}

struct Edge {
    halfedge: HalfEdge     // EITHER of the two half-edges for this edge
}
```

**How traversal works**: To iterate over all half-edges of a face, start at `face.halfedge` and follow `next` pointers until you return to the start. To iterate over all faces around a vertex, start at `vertex.halfedge` and repeatedly apply `twin.next` (or equivalently, `prev.twin`) until you return to the start.

**Why this is powerful**: Every local topological query — neighbors of a face, neighbors of a vertex, the ring of edges around a vertex, the two faces sharing an edge — resolves in O(degree) time by following pointers. No searching. No scanning.

**Boundary handling**: On a boundary edge (an edge with only one face), one half-edge has no face. Conventionally, boundary half-edges form their own loop around the boundary, with `face` set to null or to a special boundary sentinel.

The half-edge structure is used in CGAL (the Computational Geometry Algorithms Library), OpenMesh, libigl, and most serious geometry processing codebases.

**Reference**: The clearest exposition is in the CMU 15-869 Discrete Differential Geometry course wiki (Nick Sharp's writeup). The Berkeley CS184 course also has an excellent introduction. Both are freely available online.

### 3.4 Corner Table

An alternative to half-edges optimized for triangle meshes. Each triangle has three "corners," and the data structure stores, for each corner, the index of the opposite corner across the shared edge. This gives you adjacency information in a very compact form (one integer per corner). It is less flexible than half-edges (extending to non-triangle meshes is harder) but more cache-friendly and easier to implement.

### 3.5 Choosing a Data Structure

| Task | Best Structure |
|------|----------------|
| Rendering only | Face list |
| File I/O | Face list (OBJ, STL) |
| Local traversal, editing | Half-edge |
| Triangle-only processing | Corner table |
| GPU-based processing | Face list + adjacency textures |

*Exercises*: (1) Implement a half-edge data structure from scratch. Load an OBJ file, build the half-edge structure, then verify it by traversing every vertex ring and every face loop. (2) Implement the one-ring neighbor query: given a vertex, return all adjacent vertices. (3) Implement edge flip: given an edge shared by two triangles, remove it and add the other diagonal.

---

## Chapter 4: Mesh File Formats

Before you can process meshes, you need to read and write them. The two formats you will encounter most often are OBJ and STL.

### 4.1 Wavefront OBJ

OBJ is a plain-text format developed by Wavefront Technologies. Its simplicity is its greatest strength. An OBJ file looks like this:

```
# A simple triangle
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 0.5 1.0 0.0
vn 0.0 0.0 1.0
f 1//1 2//1 3//1
```

Lines beginning with `v` define vertex positions (x, y, z). Lines beginning with `vn` define vertex normals. Lines beginning with `vt` define texture coordinates (u, v). Lines beginning with `f` define faces as indices into the vertex, normal, and texture-coordinate lists.

Critical implementation details: OBJ indices start at 1, not 0. Face entries have the format `vertex_index/texture_index/normal_index`, where texture and normal indices are optional. The `//` syntax skips the texture index. OBJ files do not store units. Negative indices are relative references counting backward from the current position.

OBJ files can also contain group definitions (`g groupname`), material references (`mtllib file.mtl` and `usemtl materialname`), and smooth shading groups (`s 1` / `s off`).

**Writing an OBJ parser** is an excellent first exercise. It is simple enough to implement in an afternoon, and once you have it, you can load thousands of freely available 3D models.

### 4.2 STL (Stereolithography)

STL stores only triangle geometry — no texture coordinates, no materials, no hierarchy. Each triangle is defined by its three vertex positions and a face normal. STL comes in two flavors:

**ASCII STL**:
```
solid name
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
endsolid name
```

**Binary STL**: A compact binary encoding. 80-byte header, followed by a 4-byte triangle count, followed by each triangle as 50 bytes (12 bytes for the normal, 36 bytes for three vertices, 2 bytes for an attribute byte count typically set to zero).

STL does not share vertices between triangles — each triangle stores its own three vertex positions, leading to massive redundancy. When loading an STL, you typically need to merge coincident vertices by welding (finding vertices within a small epsilon of each other and combining them).

STL is the de facto format for 3D printing. If you work with fabrication, you will work with STL.

### 4.3 Other Formats Worth Knowing

**PLY (Polygon File Format)**: Supports both ASCII and binary. More flexible than STL — supports arbitrary per-vertex and per-face properties (colors, confidence values, etc.). Common in 3D scanning.

**glTF (GL Transmission Format)**: The "JPEG of 3D." Designed for efficient web and real-time delivery. Supports meshes, materials, textures, animations, and scene hierarchy. Binary variant (GLB) is a single self-contained file.

**OFF (Object File Format)**: Extremely simple text format. First line: `OFF`. Second line: vertex count, face count, edge count. Then vertex positions, then face vertex-index lists.

*Exercises*: (1) Write an OBJ parser and writer. (2) Write a binary STL parser. (3) Load a mesh from OBJ, convert it to STL (with vertex welding in the reverse direction), and verify the output.

---

## Chapter 5: Delaunay Triangulation

This is the most important algorithmic chapter. Delaunay triangulation is the foundation of most mesh generation algorithms, and understanding it deeply will serve you for everything that follows.

### 5.1 The Problem Statement

Given a set of n points in the plane (or in 3D), construct a triangulation — a set of non-overlapping triangles whose vertices are exactly the given points and whose union covers the convex hull of the point set — such that the Delaunay criterion is satisfied: no point lies inside the circumcircle of any triangle.

### 5.2 Why Delaunay?

The Delaunay triangulation has several optimality properties that make it uniquely useful:

**Max-min angle property**: Among all possible triangulations of a point set, the Delaunay triangulation maximizes the minimum angle. This means it avoids creating thin, needle-like triangles as much as the point set allows.

**Uniqueness**: If no four points are co-circular (a generic condition), the Delaunay triangulation is unique.

**Contains the nearest-neighbor graph**: If point A's nearest neighbor is point B, the edge AB is in the Delaunay triangulation.

**Local optimality**: A triangulation is Delaunay if and only if every interior edge is "locally Delaunay" — the edge's two adjacent triangles satisfy the empty circumcircle condition. This local characterization enables incremental construction and repair algorithms.

### 5.3 Algorithm 1: The Flip Algorithm

The simplest approach. Start with any triangulation (e.g., by sorting points and connecting them). Then scan for non-Delaunay edges and flip them.

An edge is non-Delaunay if the circumcircle of one of its adjacent triangles contains the opposite vertex of the other adjacent triangle. The **flip** operation removes this edge and replaces it with the other diagonal of the quadrilateral formed by the two triangles.

```
Before flip:          After flip:
    C                     C
   /|\                   / \
  / | \                 /   \
 /  |  \               / T1' \
A   |   D      →      A-------D
 \  |  /               \ T2' /
  \ | /                 \   /
   \|/                   \ /
    B                     B

Edge BC is non-Delaunay → flip to edge AD
```

Repeat until no non-Delaunay edges remain. The algorithm terminates because each flip strictly increases the minimum angle, and there are finitely many triangulations.

**Complexity**: O(n²) in the worst case, but often faster in practice.

### 5.4 Algorithm 2: Bowyer-Watson (Incremental Insertion)

This is the most commonly implemented Delaunay algorithm because it is simple, robust, and extends naturally to 3D.

**Steps**:

1. Create a "super-triangle" large enough to contain all input points.
2. Insert points one at a time. For each new point P:
   a. Find all triangles whose circumcircle contains P. These form a connected region called the "cavity."
   b. Delete these triangles.
   c. Connect P to every edge on the boundary of the cavity, forming new triangles.
3. After all points are inserted, remove any triangles connected to the super-triangle's vertices.

The key operation is finding the cavity — the set of triangles that violate the Delaunay condition with respect to the new point. This is done by starting from the triangle containing P (found via point location) and expanding outward, testing adjacent triangles' circumcircles.

**Complexity**: O(n log n) expected time with a good point-location strategy (e.g., walking from the most recently inserted point).

**Why this is the algorithm to implement first**: The Bowyer-Watson algorithm is fewer than 200 lines of code in most languages. It gives you a correct Delaunay triangulation. It extends to 3D (where the super-triangle becomes a super-tetrahedron and circumcircles become circumspheres). And understanding it gives you the intuition needed for Delaunay refinement (Chapter 6).

**Reference implementation guidance**: The blog post by Gorilla Sun ("Bowyer-Watson Algorithm for Delaunay Triangulation") provides a digestible walkthrough. For the authoritative treatment, see Chapter 2 of Shewchuk's "Delaunay Refinement Mesh Generation" thesis (1997), freely available online.

### 5.5 Algorithm 3: Divide and Conquer

Splits the point set in half, recursively triangulates each half, then merges the two triangulations by finding and stitching along the "merge edge." This runs in O(n log n) worst-case time. More complex to implement than Bowyer-Watson, but has better worst-case guarantees. The seminal algorithm is by Guibas and Stolfi (1985).

### 5.6 Constrained Delaunay Triangulation (CDT)

In practice, you often need specific edges to appear in the triangulation — domain boundaries, holes, material interfaces. A CDT allows you to specify a set of constraint edges that must be present. The triangulation is "as Delaunay as possible" subject to these constraints: every edge that is not a constraint edge satisfies the Delaunay criterion.

CDTs are essential for mesh generation of domains with complex boundaries. They are the starting point for Delaunay refinement algorithms.

**Key reference**: The Triangle software by Jonathan Shewchuk generates constrained Delaunay triangulations and refined meshes. It is the gold standard 2D mesh generator and is freely available.

### 5.7 3D Delaunay Triangulation

In 3D, triangles become tetrahedra, circumcircles become circumspheres, and the algorithms generalize. Bowyer-Watson works identically but in 3D. The main complication is the existence of "slivers" — tetrahedra that satisfy the Delaunay criterion but are nearly degenerate (four vertices nearly coplanar). Removing slivers requires additional techniques beyond basic Delaunay construction.

**Key software**: TetGen by Hang Si generates quality tetrahedral meshes using Delaunay refinement with sliver removal. CGAL's 3D Triangulation package is another robust option.

*Exercises*: (1) Implement Bowyer-Watson in 2D from scratch. Test on random point sets, grid point sets, and degenerate point sets (collinear points, co-circular points). (2) Visualize the triangulation and verify the empty circumcircle property. (3) Implement constrained edge insertion into your Delaunay triangulation.

---

## Chapter 6: Delaunay Refinement

Delaunay triangulation gives you a triangulation of a given point set, but the resulting triangles may still be poorly shaped if the input points are poorly distributed. Delaunay refinement solves this: it adds new points (Steiner points) to eliminate bad triangles, producing a high-quality mesh with guaranteed angle bounds.

### 6.1 The Core Idea

Start with a constrained Delaunay triangulation of the domain boundary. Identify "bad" triangles — typically, triangles with a minimum angle below some threshold (commonly 20° or 30°). For each bad triangle, insert a new point (usually the triangle's circumcenter) and re-triangulate. Repeat until no bad triangles remain.

The remarkable result (proven by Ruppert in 2D and extended by Shewchuk) is that this process terminates, and the resulting mesh has a guaranteed minimum angle bound.

### 6.2 Ruppert's Algorithm (2D)

Jim Ruppert's algorithm (1995) is the foundation of 2D Delaunay refinement. It works on planar straight line graphs (PSLGs) — domains defined by vertices and straight-line segments.

**Steps**:

1. Construct a constrained Delaunay triangulation of the input PSLG.
2. While any triangle has minimum angle < threshold:
   a. If the triangle's circumcenter encroaches on (lies within the diametral circle of) a boundary segment, split that segment at its midpoint instead.
   b. Otherwise, insert the triangle's circumcenter into the triangulation.
3. The algorithm terminates with all triangles having minimum angle ≥ threshold.

Ruppert proved termination for threshold angles up to 20.7°. In practice, 33.8° (the angle whose sine equals 1/√2) is achievable.

**Why this matters for implementation**: Ruppert's algorithm gives you a mesh generator — not just a triangulator of existing points, but a system that creates new points to guarantee mesh quality. This is what makes it practical for simulation.

**Key reference**: Ruppert, "A Delaunay Refinement Algorithm for Quality 2-Dimensional Mesh Generation," Journal of Algorithms, 1995. Also see Shewchuk's extensive treatment in his Ph.D. thesis (1997).

### 6.3 Shewchuk's Extensions

Jonathan Shewchuk extended Ruppert's algorithm with several improvements:

- Handling of small input angles (acute angles in the domain boundary).
- Extension to 3D tetrahedral meshes.
- Practical engineering decisions about data structures and numerical robustness.

His Triangle software implements these extensions and is used in production across many fields.

### 6.4 Chew's Algorithms

L. Paul Chew proposed two related algorithms. Chew's first algorithm (1989) produces meshes where all triangles are roughly the same size, with minimum angle ≥ 30°. Chew's second algorithm (1993) allows element size to vary, which is important for adaptive meshes where you want small elements in regions of interest and larger elements elsewhere.

*Exercises*: (1) Extend your Bowyer-Watson implementation with Ruppert's refinement. (2) Mesh a non-convex domain (e.g., a polygon with a hole) using CDT + refinement. (3) Implement a sizing function that controls element size across the domain.

---

## Chapter 7: Other Mesh Generation Algorithms

Delaunay-based methods are the most well-studied, but several other families of algorithms are widely used in practice.

### 7.1 Advancing Front Methods

**Idea**: Start from the boundary of the domain. Maintain a "front" — the current boundary of the meshed region. At each step, select an edge on the front and create a new triangle by introducing a new point or connecting to an existing nearby point. Advance the front inward until the entire domain is meshed.

**Strengths**: Natural boundary conformity (the mesh exactly follows the domain boundary), good element quality near boundaries, and intuitive geometric behavior.

**Weaknesses**: The front can collide with itself in concave regions, requiring complex intersection handling. The algorithm is harder to make robust than Delaunay methods. Quality in the interior (where fronts from different sides meet) can be poor.

**When to use**: When boundary conformity is paramount and the domain geometry is relatively simple. Often used in CFD pre-processing.

**Key references**: The original advancing front method by Lo (1985) and Löhner and Parikh (1988). George and Borouchaki's "Delaunay Triangulation and Meshing" (1998) covers both advancing front and Delaunay approaches.

### 7.2 Quadtree/Octree Methods

**Idea**: Recursively subdivide space using a quadtree (2D) or octree (3D) until each cell is small enough. Then triangulate each cell. Cells near the boundary are subdivided more finely to capture geometric detail.

**Strengths**: Simple to implement, naturally adaptive (small cells where needed, large cells elsewhere), parallelizable.

**Weaknesses**: Produces meshes aligned with the coordinate axes, which may not be ideal for certain simulations. Requires careful treatment of cells that straddle the domain boundary. Balancing constraints (adjacent cells should differ by at most one refinement level) add complexity.

**Key reference**: Labelle and Shewchuk, "Isosurface Stuffing: Fast Tetrahedral Meshes with Good Dihedral Angles," SIGGRAPH 2007.

### 7.3 Marching Cubes

Marching Cubes is not a mesh generation algorithm in the traditional sense — it is an **isosurface extraction** algorithm. Given a 3D scalar field f(x, y, z) and an iso-value c, it produces a triangle mesh approximating the surface where f = c.

**How it works**:

1. Divide the scalar field into a grid of cubes (voxels).
2. For each cube, classify each of the 8 corner vertices as "inside" (f > c) or "outside" (f ≤ c).
3. Since each corner is binary, there are 2⁸ = 256 possible configurations. By symmetry (rotation and reflection), these reduce to 15 unique topological cases.
4. For each configuration, a precomputed lookup table specifies which edges of the cube are intersected by the isosurface and how to triangulate them.
5. The exact position of each triangle vertex on an intersected edge is determined by linear interpolation between the edge's two endpoint scalar values.

**The 15 cases**: The lookup table encodes configurations ranging from "no corners inside" (case 0: no triangles) to "one corner inside" (case 1: one triangle cutting off that corner) to more complex configurations with multiple triangles.

**Ambiguity problem**: Some configurations (notably case 6, where two diagonally opposite corners are inside) are ambiguous — there are topologically distinct ways to triangulate them. The original Lorensen-Cline algorithm (1987) did not resolve this, leading to meshes with holes. Chernyaev (1995) and Lewiner et al. (2003) resolved this with "Marching Cubes 33," which considers all 33 topologically distinct cases.

**Where Marching Cubes is used**: Medical imaging (extracting organ surfaces from CT/MRI scans), terrain generation from heightmaps, procedural content generation, and extracting meshes from neural implicit representations (NeRFs, neural SDFs).

**Performance note**: Marching Cubes processes each cube independently, making it embarrassingly parallel and GPU-friendly. The major performance optimization is the coarse-to-fine strategy: start with a coarse grid, identify which coarse cells are near the isosurface, and only refine those.

**Key reference**: Lorensen and Cline, "Marching Cubes: A High Resolution 3D Surface Construction Algorithm," SIGGRAPH 1987. This is one of the most cited papers in computer graphics.

### 7.4 Marching Tetrahedra

A variant of Marching Cubes that decomposes each cube into tetrahedra and processes each tetrahedron. With only 4 vertices per element (2⁴ = 16 configurations, reducing to 3 cases), the lookup table is trivial and there are no ambiguities. The trade-off is more triangles in the output and less smooth surfaces.

### 7.5 Centroidal Voronoi Tessellation (CVT)

**Idea**: Place a set of generating points in the domain. Compute their Voronoi diagram. Move each generating point to the centroid (center of mass) of its Voronoi cell. Repeat until convergence. The resulting Voronoi cells are well-shaped and roughly uniform in size. Dualizing gives a high-quality Delaunay triangulation.

CVT is essentially Lloyd's algorithm (which you may know from k-means clustering — same algorithm, different context). It produces very regular meshes and is used in mesh optimization/smoothing.

**Key reference**: Du, Faber, and Gunzburger, "Centroidal Voronoi Tessellations: Applications and Algorithms," SIAM Review, 1999.

### 7.6 Paving (Quad Mesh Generation)

Paving is the advancing-front method adapted for quadrilateral meshes. It advances the boundary inward, creating rows of quads. Where fronts collide, special "seaming" operations merge them. Paving produces high-quality quad meshes and is the basis of many commercial meshing tools.

*Exercises*: (1) Implement Marching Cubes on a 3D scalar field (e.g., a sphere defined by f(x,y,z) = x²+y²+z²-1). (2) Implement Lloyd's algorithm (CVT) in 2D. Observe how the mesh quality improves with each iteration. (3) Compare the output of Delaunay refinement vs. CVT on the same domain.

---

## Chapter 8: Mesh Quality — How to Know If Your Mesh Is Good

Generating a mesh is only half the battle. You need to assess its quality, and improve it if necessary. Different applications have different quality requirements, but several metrics are universal.

### 8.1 Angle-Based Metrics

**Minimum angle**: The single most important quality metric for triangular meshes. The minimum angle across all triangles should be as large as possible. For FEA, a minimum angle below 10° is generally unacceptable. Delaunay refinement guarantees 20°+ or even 30°+.

**Maximum angle**: Large angles (close to 180°) create obtuse triangles where the circumcenter lies outside the triangle. This causes interpolation errors. Some methods bound the maximum angle to below 120° or 135°.

### 8.2 Aspect Ratio

The aspect ratio of a triangle is the ratio of its longest edge to its shortest altitude (or equivalently, its circumradius to its inradius, scaled by a constant). A perfect equilateral triangle has aspect ratio 1. Higher values indicate elongated, badly-shaped elements.

For a triangle with edge lengths a, b, c and area A:

```
Aspect Ratio = (a × b × c) / (8 × A²)    (one common definition)
```

There are multiple definitions in use (longest-to-shortest edge ratio, circumradius-to-inradius ratio, etc.), so always check which convention your software uses.

**Rules of thumb**: Aspect ratio < 5 is good, < 10 is acceptable, > 20 is problematic.

### 8.3 Skewness

Skewness measures the deviation of an element from its ideal shape. For a triangle, the ideal shape is equilateral. Skewness is defined as:

```
Skewness = (θ_max - θ_ideal) / (180° - θ_ideal)
```

where θ_max is the largest angle in the element and θ_ideal is 60° for triangles (90° for quads). A skewness of 0 is ideal; values above 0.75 indicate poor quality; values above 0.95 indicate nearly degenerate elements.

### 8.4 Jacobian and Jacobian Ratio

For higher-order elements (quadratic triangles, hexahedra, etc.), the Jacobian matrix maps from the element's reference (ideal) coordinates to its physical coordinates. The determinant of this Jacobian should be positive everywhere inside the element. If it goes negative, the element is inverted (inside-out), which is catastrophic for simulation.

The Jacobian ratio is the minimum Jacobian determinant divided by the maximum Jacobian determinant, evaluated at integration points. Ideally it is 1.0; values below 0.5 are concerning; negative values indicate inversion.

### 8.5 Element Size Distribution

For simulation, you often want element sizes to vary smoothly — small elements in regions of interest (near boundaries, around features, in high-gradient areas) and larger elements elsewhere. Abrupt size transitions cause interpolation errors. A good mesh has a smooth size gradation, typically with adjacent elements differing in size by no more than a factor of 2.

### 8.6 Mesh Smoothing (Post-Processing)

After generating a mesh, you can improve quality by adjusting vertex positions without changing connectivity. The most common method is **Laplacian smoothing**: move each vertex to the centroid of its neighbors. This tends to produce more regular elements but can invert elements near boundaries or concavities. **Smart Laplacian smoothing** checks quality before and after each move, only accepting moves that improve quality. **Optimization-based smoothing** formulates quality improvement as an optimization problem, minimizing a distortion metric.

*Exercises*: (1) Implement minimum angle, aspect ratio, and skewness computation for a triangle mesh. (2) Generate a histogram of element quality for a mesh. (3) Implement Laplacian smoothing and measure quality before and after.

---

## Chapter 9: Subdivision Surfaces and Mesh Refinement

Subdivision surfaces are a technique for generating smooth surfaces from coarse polygon meshes. They are foundational in animation (Pixar uses Catmull-Clark subdivision for nearly all their characters) and provide a bridge between the discrete world of meshes and the continuous world of smooth surfaces.

### 9.1 The Subdivision Idea

Start with a coarse "control mesh." Apply a refinement rule that:
1. Adds new vertices (on edges and/or face centroids).
2. Adjusts positions of old and new vertices using weighted averages of their neighbors.
3. Creates new, smaller faces.

Repeat. After a few iterations, the mesh converges to a smooth limit surface. The control mesh acts as a "skeleton" that defines the shape; the subdivision rules define how the smooth surface wraps around it.

### 9.2 Loop Subdivision (for Triangle Meshes)

Developed by Charles Loop (1987). Each triangle is split into four smaller triangles by inserting a new vertex at the midpoint of each edge. Vertex positions are updated using weighted averages.

**New edge vertex** (on edge connecting vertices v₁ and v₂, with opposite vertices v₃ and v₄):
```
v_new = (3/8)(v₁ + v₂) + (1/8)(v₃ + v₄)
```

**Updated old vertex** (with n neighbors q₁, ..., qₙ):
```
v_updated = (1 - n×β) × v_old + β × (q₁ + q₂ + ... + qₙ)
```
where β = (1/n)(5/8 - (3/8 + (1/4)cos(2π/n))²).

Loop subdivision produces C² continuous surfaces everywhere except at "extraordinary vertices" (vertices with valence ≠ 6), where continuity is C¹. The limit surface is a quartic box spline surface away from extraordinary vertices.

### 9.3 Catmull-Clark Subdivision (for Quad Meshes)

Developed by Edwin Catmull and Jim Clark (1978). This is the most widely used subdivision scheme in the film industry.

**Steps per iteration**:

1. **Face points**: For each face, compute a new vertex at the centroid of the face's vertices.
2. **Edge points**: For each edge, compute a new vertex as the average of the edge's two endpoints and the two adjacent face points.
3. **Updated vertex points**: For each original vertex P with n neighboring face points F₁...Fₙ and n edge midpoints R₁...Rₙ:
```
P_new = (F_avg + 2×R_avg + (n-3)×P) / n
```
where F_avg is the average of face points and R_avg is the average of edge midpoints.
4. Connect each face point to its edge points, and each edge point to the updated vertex points.

After one iteration of Catmull-Clark subdivision on any polyhedron, every face becomes a quadrilateral. Subsequent iterations refine the quad mesh. The limit surface is a bicubic B-spline surface away from extraordinary vertices (vertices with valence ≠ 4).

### 9.4 Other Subdivision Schemes

**Doo-Sabin subdivision** (1978): Dual to Catmull-Clark. Generates new vertices inside each face, on each edge, and at each vertex. Produces a face for each old vertex, each old edge, and each old face. Converges to biquadratic B-spline surfaces.

**√3 subdivision** (Kobbelt, 2000): For triangle meshes. Inserts a vertex at each face center and connects it to the face's vertices, then flips all old edges. Refines more slowly than Loop (each iteration multiplies the face count by 3, not 4), which allows finer control over mesh density.

**Butterfly subdivision** (Dyn, Levine, and Gregory, 1990): An *interpolating* scheme — old vertices do not move. This preserves the original geometry while smoothing, which is useful when the original vertex positions carry data (e.g., measured points from a 3D scan).

### 9.5 Adaptive Subdivision

You do not always need to subdivide uniformly. Adaptive subdivision refines only in regions where more detail is needed — near high curvature, near features, or near the camera. This dramatically reduces face counts while maintaining visual quality. Feature-adaptive subdivision (used in OpenSubdiv) subdivides fully only near extraordinary vertices and features, using hardware tessellation for regular patches.

*Exercises*: (1) Implement Loop subdivision. Start with a tetrahedron, subdivide three times, and observe it converging to a sphere. (2) Implement Catmull-Clark on a cube mesh. (3) Compare the limit surfaces of Loop and Butterfly subdivision on the same input mesh.

---

## Chapter 10: Neural and Learned Mesh Generation

Given your ML background, this chapter bridges the classical algorithms above with modern deep-learning approaches to mesh generation. This is a rapidly evolving field, so this section focuses on fundamental approaches and architectural paradigms rather than specific model versions.

### 10.1 The Representation Problem

Neural networks operate on tensors — fixed-size, regular grids. Meshes are irregular: variable numbers of vertices, variable connectivity, non-Euclidean structure. The core challenge is representing mesh data in a form neural networks can consume and produce.

**Major approaches**:

1. **Voxel grids**: Discretize 3D space into a regular grid. Each voxel is occupied or empty. This is the simplest representation but scales cubically with resolution (a 256³ grid has 16 million voxels). Marching Cubes extracts the final mesh.

2. **Point clouds**: Unordered sets of points. No connectivity information. Networks like PointNet and PointNet++ operate directly on point clouds. To get a mesh, you need surface reconstruction as a post-processing step (e.g., Poisson reconstruction, ball-pivoting).

3. **Implicit functions (Neural SDFs/Occupancy Networks)**: The network learns a function f(x, y, z) that returns the signed distance to the surface (or an occupancy probability). The mesh is extracted as the zero-level-set using Marching Cubes. This decouples the network output (a continuous function) from the discrete mesh, allowing arbitrary resolution at extraction time.

4. **Mesh deformation**: Start with a template mesh (e.g., a sphere) and learn per-vertex displacements. This preserves topology (same connectivity as the template) but limits the shapes that can be represented.

5. **Autoregressive mesh generation**: Directly output vertices and faces as sequences of tokens, similar to language models generating text. Recent work (MeshGPT, MeshXL, EdgeRunner, MeshAnything) tokenizes mesh geometry and connectivity and generates them autoregressively.

### 10.2 Neural Radiance Fields (NeRF) and Mesh Extraction

NeRFs represent scenes as a continuous volumetric function mapping 3D position and viewing direction to color and density. They produce photorealistic novel views but do not directly output meshes. Extracting a mesh from a NeRF typically involves:

1. Querying the density field on a 3D grid.
2. Running Marching Cubes on the density field to get a surface mesh.
3. Optionally baking the appearance (colors/textures) onto the mesh.

The quality of NeRF-to-mesh conversion depends heavily on the quality of the density field. Methods like NeuS and VolSDF replace density with a signed distance function, which produces cleaner surfaces for mesh extraction.

### 10.3 3D Gaussian Splatting and Meshes

3D Gaussian Splatting represents scenes as collections of anisotropic Gaussians rather than as meshes or volume fields. Converting Gaussian splats to meshes is an active research area. Current approaches typically involve rasterizing the Gaussians onto a grid and running Marching Cubes, or using Poisson surface reconstruction on the Gaussian centers.

### 10.4 Diffusion Models for 3D Generation

The success of diffusion models for 2D image generation has inspired their application to 3D. Current approaches include:

**Multi-view diffusion**: Generate multiple 2D views of an object, then reconstruct the 3D mesh using multi-view stereo or a feed-forward reconstruction network.

**3D-native diffusion**: Diffuse directly in 3D space — on point clouds, on voxel grids, or on latent 3D representations. Notable examples include LION (latent point diffusion) and various SDF-based diffusion models.

**Score distillation**: Use a pretrained 2D diffusion model as a critic to optimize a 3D representation (DreamFusion, Score Jacobian Chaining). The 3D representation is typically a NeRF or mesh, optimized so that its 2D renderings look plausible to the diffusion model.

### 10.5 Autoregressive Mesh Generation

The most recent paradigm treats mesh generation as a sequence modeling problem. The mesh is serialized into a sequence of tokens (vertex coordinates, face definitions), and a transformer generates this sequence autoregressively.

**MeshGPT** (Siddiqui et al., 2023): Tokenizes triangles using a VQ-VAE, then generates token sequences with a GPT-style transformer. Can generate meshes conditioned on point clouds, images, or text.

**MeshAnything** (Chen et al., 2024): Converts arbitrary 3D representations (point clouds, meshes, NeRFs) into artist-quality meshes with clean topology, using autoregressive generation.

The core insight is that artist-created meshes have structure — regular edge flows, quad-dominant topology, clean edge loops — that is learnable. Autoregressive models can capture these patterns in a way that purely geometric algorithms cannot.

### 10.6 Geometric Deep Learning on Meshes

If your interest is not generating meshes but processing them with neural networks, the relevant tools are:

**MeshCNN** (Hanocka et al., 2019): Defines convolution, pooling, and unpooling operations on mesh edges. Edge collapse is used as a pooling operation.

**DiffusionNet** (Sharp et al., 2022): A mesh-based neural network built on the Laplace-Beltrami operator. Architecture-agnostic to mesh resolution and discretization.

**PyTorch3D** (Ravi et al., 2020): Facebook's differentiable rendering and mesh processing library. Supports differentiable mesh operations, making it possible to optimize mesh geometry via gradient descent.

*Exercises*: (1) Use PyTorch3D to load a mesh, compute the Laplacian, and perform mesh smoothing via gradient descent. (2) Implement a simple neural SDF: train an MLP to predict signed distance from point coordinates, then extract the mesh with Marching Cubes. (3) Read the MeshGPT paper and understand how mesh tokenization works.

---

## Chapter 11: Curated Reading List

This is not a dump of every paper ever written. This is a curated, opinionated list of the readings that will teach you the most, organized by topic and difficulty.

### Textbooks (Start Here)

**"Polygon Mesh Processing"** by Mario Botsch, Leif Kobbelt, Mark Pauly, Pierre Alliez, and Bruno Lévy (A K Peters, 2010). *This is your primary textbook.* It covers mesh data structures, surface smoothing, parameterization, remeshing, simplification, and deformation. Written for implementers, with pseudocode and clear explanations. Available as a free PDF from the authors' websites.

**"Delaunay Mesh Generation"** by Siu-Wing Cheng, Tamal K. Dey, and Jonathan Shewchuk (CRC Press, 2012). The definitive treatment of Delaunay-based mesh generation algorithms. Covers 2D and 3D, surface and volume meshes. Rigorous but accessible. This is the book to read after you have implemented a basic Delaunay triangulation and want to understand the theory deeply.

**"Computational Geometry: Algorithms and Applications"** by de Berg, Cheong, van Kreveld, and Overmars (Springer, 3rd edition, 2008). The standard computational geometry textbook. Chapters on Delaunay triangulations, Voronoi diagrams, convex hulls, and geometric data structures provide the algorithmic foundations needed for mesh generation.

**"Geometry and Topology for Mesh Generation"** by Herbert Edelsbrunner (Cambridge University Press, 2001). A more mathematical treatment connecting mesh generation to topology. Good if you want to understand the theoretical underpinnings of why algorithms work.

### Foundational Papers

**Lorensen and Cline, "Marching Cubes: A High Resolution 3D Surface Construction Algorithm," SIGGRAPH 1987.** The paper that introduced Marching Cubes. One of the most cited papers in computer graphics. Read for historical context and the elegant simplicity of the lookup-table approach.

**Ruppert, "A Delaunay Refinement Algorithm for Quality 2-Dimensional Mesh Generation," Journal of Algorithms, 1995.** The paper that proved Delaunay refinement terminates with guaranteed angle bounds. The algorithm is practical and the proof is beautiful.

**Shewchuk, "Delaunay Refinement Mesh Generation," Ph.D. thesis, Carnegie Mellon University, 1997.** Not a paper but a thesis, and one of the best-written technical documents in the field. Freely available at: http://www.cs.cmu.edu/~quake-papers/delaunay-refinement.pdf. Covers theory, algorithms, implementation, and data structures. If you read one document beyond this guide, read this.

**Shewchuk, "What Is a Good Linear Finite Element? Interpolation, Conditioning, Anisotropy, and Quality Measures," 2002.** Explains *why* mesh quality matters from the finite element perspective. Connects geometric quality metrics to numerical accuracy.

**Catmull and Clark, "Recursively Generated B-spline Surfaces on Arbitrary Topological Meshes," Computer-Aided Design, 1978.** The original Catmull-Clark subdivision paper. Short, clear, and historically important.

**Loop, "Smooth Subdivision Surfaces Based on Triangles," Master's thesis, University of Utah, 1987.** Introduces Loop subdivision for triangle meshes.

### Implementation-Oriented Resources

**Shewchuk's Triangle software**: https://www.cs.cmu.edu/~quake/triangle.html — The reference 2D mesh generator. The source code is a masterclass in robust geometric computing.

**CGAL (Computational Geometry Algorithms Library)**: https://www.cgal.org — Industrial-strength C++ library with implementations of Delaunay triangulation (2D and 3D), mesh generation, mesh processing, and geometric predicates.

**libigl**: https://libigl.github.io — A lightweight C++ geometry processing library with a focus on research prototyping. Excellent for quickly implementing mesh algorithms.

**OpenMesh**: https://www.graphics.rwth-aachen.de/software/openmesh/ — A C++ half-edge mesh library from RWTH Aachen. Clean API, well-documented.

**PyMeshLab / MeshLab**: https://www.meshlab.net — GUI + Python bindings for mesh processing. Good for visualization and rapid prototyping.

**Gorilla Sun blog, "Bowyer-Watson Algorithm for Delaunay Triangulation"**: https://www.gorillasun.de/blog/bowyer-watson-algorithm-for-delaunay-triangulation/ — One of the best implementation tutorials for Bowyer-Watson. Code is readable and well-explained.

**CMU 15-869 Discrete Differential Geometry (Keenan Crane)**: Course notes, slides, and assignments freely available. The half-edge data structure writeup (Nick Sharp) is the clearest explanation available anywhere. https://brickisland.net/DDGSpring2016/

**Berkeley CS184/284A**: The half-edge data structure introduction is available at https://cs184.eecs.berkeley.edu/sp24/docs/half-edge-intro — clear, concise, with diagrams.

**Scratchapixel**: https://www.scratchapixel.com — Excellent tutorials on 3D rendering fundamentals, including mesh file format parsing.

### Advanced / Specialist References

**"Handbook of Discrete and Computational Geometry" (3rd edition)** edited by Goodman, O'Rourke, and Tóth. Chapter on "Triangulations and Mesh Generation" by Bern, Shewchuk, and Amenta is available as a free PDF and provides a comprehensive survey.

**Shewchuk, "Robust Adaptive Floating-Point Geometric Predicates," 1997.** If you implement Delaunay triangulation, you will eventually encounter floating-point precision bugs. This paper explains the problem and provides the solution (exact arithmetic predicates). The code is at https://www.cs.cmu.edu/~quake/robust.html.

**Du, Faber, and Gunzburger, "Centroidal Voronoi Tessellations: Applications and Algorithms," SIAM Review, 1999.** The foundational paper on CVT for mesh optimization.

**Labelle and Shewchuk, "Isosurface Stuffing: Fast Tetrahedral Meshes with Good Dihedral Angles," SIGGRAPH 2007.** An octree-based tetrahedral meshing algorithm with provable quality guarantees.

**Garland and Heckbert, "Surface Simplification Using Quadric Error Metrics," SIGGRAPH 1997.** The seminal paper on mesh simplification. Introduces the Quadric Error Metric (QEM), which is still the standard approach. Essential reading if you need to reduce mesh complexity.

### Neural / Learned Approaches

**Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis," ECCV 2020.** The paper that launched the NeRF revolution. Not directly about meshes, but critical context for understanding modern 3D generation.

**Park et al., "DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation," CVPR 2019.** Represents shapes as neural implicit functions. Mesh extraction via Marching Cubes.

**Siddiqui et al., "MeshGPT: Generating Triangle Meshes with Decoder-Only Transformers," CVPR 2024.** Autoregressive mesh generation using a VQ-VAE + transformer architecture.

**Hanocka et al., "MeshCNN: A Network with an Edge," SIGGRAPH 2019.** Defines convolution on mesh edges. Important for understanding how to apply deep learning to mesh data.

**Sharp et al., "DiffusionNet: Discretization Agnostic Learning on Surfaces," ACM TOG, 2022.** A mesh neural network that works regardless of the mesh's triangulation quality or density. Uses the Laplace-Beltrami operator as a foundation.

---

## Appendix A: Quick Reference — Algorithm Selection

| I need to... | Algorithm |
|---|---|
| Triangulate a 2D point set | Bowyer-Watson (Delaunay) |
| Mesh a 2D domain with quality guarantees | Delaunay refinement (Ruppert/Shewchuk) |
| Extract a surface from volumetric data | Marching Cubes |
| Generate a tetrahedral volume mesh | 3D Delaunay refinement (TetGen) |
| Smooth a coarse mesh into a high-res one | Subdivision (Loop for tris, Catmull-Clark for quads) |
| Reduce the number of triangles | Quadric Error Metric simplification |
| Improve mesh quality after generation | Laplacian smoothing, optimization-based smoothing, CVT |
| Generate a mesh from a neural implicit | Marching Cubes on the learned SDF |
| Generate an artist-quality mesh from a 3D scan | MeshAnything / autoregressive approaches |

## Appendix B: Software Ecosystem Map

| Software | Language | What It Does |
|---|---|---|
| Triangle | C | 2D Delaunay mesh generation (the gold standard) |
| TetGen | C++ | 3D tetrahedral mesh generation |
| CGAL | C++ | Comprehensive computational geometry library |
| libigl | C++ (Python bindings) | Research geometry processing |
| OpenMesh | C++ | Half-edge mesh library |
| PyMeshLab | Python | Mesh processing with MeshLab backend |
| Trimesh | Python | Simple mesh loading/processing |
| PyTorch3D | Python | Differentiable mesh processing for ML |
| Open3D | C++/Python | 3D data processing (point clouds + meshes) |
| Blender (Python API) | Python | Full 3D modeling, scripting, export |

---

*This document was compiled as a comprehensive study guide for understanding and implementing mesh generation from first principles. Start with Chapter 1, implement as you go, and refer to the reading list when you want depth on a particular topic. The field is deep, but the fundamentals are finite — master the half-edge data structure, Bowyer-Watson, and Marching Cubes, and you can build anything.*