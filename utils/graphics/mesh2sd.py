import os
import torch
import torch.nn.functional as F
from glob import glob
from torch.autograd import Function
from torch.utils.cpp_extension import load

EXT  = os.path.join(os.path.dirname(__file__), "ext")
NAME = "mesh2sd"

src_list = []
src_list += glob(os.path.join(EXT, NAME, "*.c"))
src_list += glob(os.path.join(EXT, NAME, "*.cpp"))
src_list += glob(os.path.join(EXT, NAME, "*.cu"))

mesh2sd_impl = load(name=NAME, sources=src_list, verbose=True)

class Mesh2SignedDistanceFunction(Function):
    @staticmethod
    def forward(ctx, faces, query_pnt):

        # bs    = mesh_ver.size(0)
        # nf    = mesh_tri.size(1)

        # query = query_pnt.to(faces.dtype)
        query = query_pnt

        ctx.save_for_backward(faces, query_pnt)

        sdist, index, weight = mesh2sd_impl.sd_query_forward(faces, query)
        return sdist, index

    @staticmethod
    def backward(ctx, *args):

        faces, query_pnt = ctx.saved_tensors

        eps = 1e-4
        deltas = torch.as_tensor([
            [ 1, -1, -1],
            [-1, -1,  1],
            [-1,  1, -1],
            [ 1,  1,  1],
        ]).to(query_pnt.device, non_blocking=True)
        deltas = deltas.reshape([1]*(query_pnt.dim()-1)+[4,3])

        query  = query_pnt.unsqueeze(-2) + eps*deltas

        d, i = Mesh2SignedDistanceFunction.apply(faces, query.reshape(faces.size(0), -1, 3))

        shape = list(query.shape[:-1]) + [1]

        d = d.reshape(*shape)                # ...,1

        g = torch.sum(d*deltas, dim=-2)      # ...,4,3 -> ...,3
        g = F.normalize(g, dim=-1)

        return None, g

def barycentric_coords(a, b, c, q, eps=1e-16):
    '''
    Compute Barycentric coordinates of q projected on the triangle spanned by the vertices a, b, and c
    '''

    v1 = b-a
    v2 = c-a
    n = torch.cross(v1, v2)
    n_dot_n = torch.sum(n*n, dim=-1)
    n_dot_n[n_dot_n < eps] = 1.0
    
    w = q - a
    gamma = torch.sum(torch.cross(v1, w)*n, dim=-1)/n_dot_n
    beta = torch.sum(torch.cross(w, v2)*n, dim=-1)/n_dot_n
    alpha = 1.0-beta-gamma
    return torch.stack((alpha, beta, gamma), dim=-1)

def point_mesh_distance(points, ver, tri, signed=False):
    import kaolin
    
    from kaolin.ops.mesh import index_vertices_by_faces
    from kaolin.metrics.trianglemesh import point_to_mesh_distance

    batch_size, num_points, _ = points.shape
    mesh_face_vertices = index_vertices_by_faces(ver, tri) # (batch_size, num_faces, num_vertices, 3)

    # Get closest mesh triangles for each point
    distance, index, dist_type = point_to_mesh_distance(points, mesh_face_vertices)

    distance = (distance + 1e-8).sqrt()

    # closest_triangle_vertices = torch.gather(mesh_face_vertices, dim=1, index=face_idx[:, :, None, None].expand(-1, -1, 3, 3)).view(batch_size, num_points, 3, 3)

    # # Compute barycentric embedding of every point into the closest triangle
    # a, b, c = closest_triangle_vertices[:, :, 0, :], closest_triangle_vertices[:, :, 1, :], closest_triangle_vertices[:, :, 2, :]   # (batch_size, num_points, 3)
    # bcoords = barycentric_coords(a, b, c, points)    # (batch_size, num_points, 3)

    # closest_points = torch.multiply(a, bcoords[:, :, 0].unsqueeze(-1)) + \
    #                 torch.multiply(b, bcoords[:, :, 1].unsqueeze(-1)) + \
    #                 torch.multiply(c, bcoords[:, :, 2].unsqueeze(-1))   # (batch_size, num_points, 3)
    # square_distances = (closest_points - points).square().sum(-1)

    if signed:
        is_inside = kaolin.ops.mesh.check_sign(ver, tri, points)
        return torch.where(is_inside, -distance, distance)
    else:
        return distance

if __name__ == "__main__":

    def pack2faces(ver, tri):
        if ver.dim() == 2:
            return ver[tri.flatten()].reshape(tri.size(0), 3, 3)
        elif ver.dim() == 3:
            f = torch.gather(ver, 1, tri.flatten(1).unsqueeze(-1).expand(-1,-1,3))
            f = f.reshape(ver.size(0), tri.size(1), 3, 3)
            return f


    scene_v = torch.FloatTensor([
        [0, 0, 0],
        [0, 0, 3],
        [1, 0, 0],
    ])
    scene_t = torch.LongTensor([[0,1,2]])

    faces_t = pack2faces(scene_v, scene_t)

    # query_p = torch.FloatTensor([
    #     [0, 0, 0],
    #     [1, 0, 0],
    #     [0, 0, 1],
    # ])
    # query_p[:, 1] += 0.1

    normal = torch.cross(
        scene_v[ scene_t[:,1] ] - scene_v[ scene_t[:, 0] ], 
        scene_v[ scene_t[:,2] ] - scene_v[ scene_t[:, 0] ], dim=-1)
    normal = torch.nn.functional.normalize(normal, dim=-1)
    scene_v, scene_t, normal = map(lambda x:x.unsqueeze(0), 
        [scene_v, scene_t, normal])
    faces_t = faces_t.unsqueeze(0)
    print(normal)

    normal = torch.FloatTensor([0, 1, 0]).reshape(1,1,3)
    
    # inside
    for i in range(0):
        w2 = torch.rand(1, 2)
        w2 = torch.where( w2.sum(dim=-1, keepdim=True) > 1, 1 - w2, w2)
        w3 = torch.cat([w2, 1-w2.sum(dim=-1, keepdim=True)], dim=-1)
        query_b = torch.einsum("nk,bkc->bnc", w3, scene_v)
        print(w3, query_b)
        # query_b = scene_v
        offset_d= torch.randn((query_b.size(0),query_b.size(1)))*0.1
        query_p = query_b + offset_d[...,None] * normal[:,:1,:]
    
        s_dist = Mesh2SignedDistanceFunction.apply(scene_v, scene_t, query_p)
        correct= torch.allclose(s_dist, offset_d)
        print(correct, offset_d.flatten(), s_dist.flatten())

    succ = []
    for i in range(1):
        # w1 = torch.rand(1)
        # w3 = torch.stack([w1, -w1, torch.ones(w1.size(0))], dim=-1)
        w2 = torch.rand((1,2))*2-1
        w3 = torch.cat([-w2, torch.ones(w2.size(0),1)+w2.sum(dim=-1,keepdim=True)], dim=-1)
        ri = torch.stack([torch.randperm(3) for bi in range(w3.size(0))], dim=0)
        w3 = torch.gather(w3, 1, ri)
        query_b = torch.einsum("nk,bkc->bnc", w3, scene_v)
        # query_b = scene_v
        offset_d= torch.randn((query_b.size(0),query_b.size(1), 1)) * normal[:,:1,:] * 0.2
        query_p = query_b + offset_d
        print(w3, query_b, query_p)

        q2p = query_b[:,None] - scene_v[:,:,None] # bkc, bnc -> bnkc
        q2p = torch.linalg.norm(q2p, dim=-1)
        q2p = torch.min(q2p, dim=1).values

        sign = 1#torch.sign((offset_d * normal).sum(-1))
        dist = sign* torch.sqrt( torch.linalg.norm(offset_d, dim=-1).pow(2) + q2p.pow(2))

        faces = pack2faces(scene_v, scene_t[:,:1])
        s_dist = Mesh2SignedDistanceFunction.apply(faces, query_p.float())[0]
        correct= torch.allclose(s_dist, dist)
        print(correct, dist.flatten(), s_dist.flatten())

        succ.append(correct)
    print(succ)


    scene_v = torch.FloatTensor([
        [0, 0, 0],
        [0, 0, 2],
        [1, 0, 0],
        [1, 0, 1],
    ])
    scene_t = torch.LongTensor([[0,1,2], [2,1,3]])
    scene_v, scene_t = map(lambda x:x.unsqueeze(0), 
        [scene_v, scene_t])

    # inside
    for i in range(1):
        w2 = torch.rand((10,2))#*3-1.5
        w3 = torch.cat([w2[:,:1], torch.zeros(w2.size(0),1), w2[:,1:]], dim=-1)
        query_b = w3.unsqueeze(0)
        offset_d= torch.randn((query_b.size(0),query_b.size(1), 1)) * normal[:,:1,:] * 0.2
        query_p = query_b + offset_d
        print(w3, query_b, query_p)

        q2p = query_b[:,None] - scene_v[:,:,None] # bkc, bnc -> bnkc
        q2p = torch.linalg.norm(q2p, dim=-1)
        q2p = torch.min(q2p, dim=1).values

        sign = 1#torch.sign((offset_d * normal).sum(-1))
        dist = sign* query_p[:,:,1].abs()
    
        faces = pack2faces(scene_v, scene_t)
        s_dist = Mesh2SignedDistanceFunction.apply(faces, query_p.float())[0]
        correct= torch.allclose(s_dist, dist)
        print(correct, dist.flatten(), s_dist.flatten())


    dev = torch.device("cuda", 0)
    faces = pack2faces(scene_v, scene_t)
    c_dist, c_indx = Mesh2SignedDistanceFunction.apply(faces, query_p.float())
    d_dist, d_indx = Mesh2SignedDistanceFunction.apply(faces.to(dev), query_p.float().to(dev))

    print("dist close", torch.allclose(c_dist, d_dist.cpu()))
    print("indx close", torch.allclose(c_indx, d_indx.cpu()))

    print(c_dist, d_dist.cpu())
    print(c_indx, d_indx.cpu())