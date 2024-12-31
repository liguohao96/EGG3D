import numpy as np
import mesh

print(dir(mesh))

gp = np.random.randn(4, 3).astype(np.float32)
gf = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
tc = np.ones_like(gp).astype(np.uint8)
uv = np.random.randn(4, 2).astype(np.float32)
print(gp.shape, tc.shape)

d = {
    "geometry_vertex": gp,
    "texture_file": "1.png"
}
mesh.save_ply_file("test0.ply", d, False)
mesh.save_ply_file("test0b.ply", d, True)

# d["texture_color"] = tc
# mesh.save_ply_file("test1.ply", d, False)
# mesh.save_ply_file("test1b.ply", d, True)

d["geometry_face"] = gf
mesh.save_ply_file("test2.ply", d, False)
mesh.save_ply_file("test2b.ply", d, True)

d["texture_coord"] = uv
mesh.save_ply_file("test3.ply", d, False)
mesh.save_ply_file("test3b.ply", d, True)
