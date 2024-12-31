import os
from glob import glob
import numpy as np
# from .ext import mesh as mesh_io_c

from torch.utils.cpp_extension import load

EXT  = os.path.join(os.path.dirname(__file__), "ext")
NAME = "mesh"

src_list = []
src_list += glob(os.path.join(EXT, NAME, "*.c"))
src_list += glob(os.path.join(EXT, NAME, "*.cpp"))
inc_dirs = [os.path.join(EXT, NAME)]

mesh_io_c = load(name=NAME, sources=src_list,
    extra_include_paths=inc_dirs,
    extra_cflags=[f"-D EXTENSION_NAME={NAME}"])

def save_mesh_as_ply(ply_f, mesh, binary=True, verbose=True):
    '''
    geometry_vertex
    geometry_face
    texture_coord
    texture_file
    '''

    mesh["geometry_vertex"] = mesh["geometry_vertex"].astype(np.float32)
    if "geometry_face" in mesh:
        mesh["geometry_face"] = mesh["geometry_face"].astype(np.int32)

    vertex  = mesh["geometry_vertex"]
    tri     = mesh.get("geometry_face", None)

    num_ver = vertex.shape[0]
    if "texture_coord" in mesh:
        uv = mesh["texture_coord"]
        if uv.shape[0] == num_ver:
            # uv per vertex
            assert uv.shape[1] == 2, f"uv:{uv.shape} vertex:{vertex.shape} tri:{tri.shape}"
            # uv = np.stack([uv[tri[:, 0]] for i in range(3)], axis=1)
        else:
            assert uv.shape[0] == tri.shape[0], f"uv:{uv.shape} vertex:{vertex.shape} tri:{tri.shape}"
            assert uv.shape[1] == 3, f"uv:{uv.shape} vertex:{vertex.shape} tri:{tri.shape}"
            assert uv.shape[2] == 2, f"uv:{uv.shape} vertex:{vertex.shape} tri:{tri.shape}"
            uv = uv.reshape(tri.shape[0], -1)
        mesh["texture_coord"] = uv.astype(np.float32)

        if "texture_color" in mesh:
            del mesh["texture_color"]
    
    if verbose:
        for k, v in mesh.items():
            if isinstance(v, np.ndarray):
                print(f"{k} {v.shape}")
    ret = mesh_io_c.save_ply_file(ply_f, mesh, binary)
    if verbose:
        print(f"[{ret}] {ply_f}")

class OBJReader(object):
    def __init__(self):
        super().__init__()
    
    @classmethod
    def load_obj(cls, fp):
        vs, t_vs = [], []
        vt, t_vt = [], []
        vn, t_vn = [], []

        for ln, lc in enumerate(fp):
            line            = lc.strip()
            splitted_line   = line.split()
            line_type       = splitted_line[0] if len(splitted_line) != 0 else None
            
            splited_num     = 4
            if not splitted_line:
                continue
            elif line_type == 'v':
                vs.append([float(v) for v in splitted_line[1:4]])
            elif line_type == 'vt':
                splited_num = 3
                vt.append([float(v) for v in splitted_line[1:3]])
            elif line_type == 'vn':
                vn.append([float(v) for v in splitted_line[1:4]])
            elif line_type == 'f':
                f_len      = len(splitted_line[1].split('/'))
                
                f_arr      = [list() for i in range(3)]
                
                for c in splitted_line[1:]:
                    fs     = c.split('/')
                    for i, fi in enumerate(fs):
                        if len(fi) > 0:
                            fi = int(fi)
                            f_arr[i].append(fi)
                
                s_ids      = f_arr[0]
                t_ids      = f_arr[1]
                n_ids      = f_arr[2]

                
                tri_ids    = [(ind - 1) if (ind > 0) else (len(vs) + ind)
                                   for ind in s_ids]
                tex_ids    = [(ind - 1) if (ind > 0) else (len(vt) + ind)
                                   for ind in t_ids]
                nrm_ids    = [(ind - 1) if (ind > 0) else (len(vn) + ind)
                                   for ind in n_ids]
                assert len(tex_ids) in [0, 3], f"line: {ln}, got {len(tex_ids)} tex"
                t_vs.append(tri_ids)
                t_vt.append(tex_ids)
                if len(nrm_ids) > 0:
                    t_vn.append(nrm_ids)
            else:
                continue
            assert len(splitted_line) == splited_num, f"Error with line {ln+1}: '{lc}'"
            
        vs    = np.asarray(vs)
        vt    = np.asarray(vt)
        vn    = np.asarray(vn)
        t_vs  = np.asarray(t_vs, dtype=int)
        t_vt  = np.asarray(t_vt, dtype=int)
        t_vn  = np.asarray(t_vn, dtype=int)
        return vs, vt, vn, t_vs, t_vt, t_vn

def load_mesh(mesh_f, filters=None):
    try:
        import pymeshlab
    except Exception as ex:
        import os
        if os.path.splitext(mesh_f)[-1] == ".obj":
            with open(mesh_f) as f:
                ret = OBJReader.load_obj(f)
            return ret

        # import sysconfig
        # LIBDIR = sysconfig.get_config_var("LIBDIR")
        # import ctypes
        # LIBQT5 = os.path.join(LIBDIR, "libQt5Core.so.5")
        # if os.path.exists(LIBQT5):
        #     ctypes.CDLL(LIBQT5)
        #     ctypes.CDLL(os.path.join(LIBDIR, "libQt5Xml.so.5"))
        #     ctypes.CDLL(os.path.join(LIBDIR, "libQt5Gui.so.5"))
        #     ctypes.CDLL(os.path.join(LIBDIR, "libQt5Widgets.so.5"))
        #     ctypes.CDLL(os.path.join(LIBDIR, "libQt5OpenGL.so.5"))
        # import pymeshlab

    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(mesh_f)

    filters = filters if filters is not None else []
    for f in filters:
        ms.apply_filter(f[0], **f[1])

    mesh = ms.current_mesh()
    # try:
    #     mesh.wedge_tex_coord_matrix()
    #     ms.apply_filter("compute_texcoord_transfer_wedge_to_vertex")
    #     mesh = ms.current_mesh()
    # except Exception as ex:
    #     pass

    vs = mesh.vertex_matrix()
    try:
        vt = mesh.vertex_tex_coord_matrix()
    except Exception as ex:
        vt = np.asarray([], dtype=float)
    vn = None

    try:
        t_vs = mesh.face_matrix()
    except Exception as ex:
        t_vs = np.asarray([], dtype=int)
    try:
        t_vt = mesh.face_matrix()
    except Exception as ex:
        t_vt = np.asarray([], dtype=int)
    t_vn = None

    return vs, vt, vn, t_vs, t_vt, t_vn

def save_mesh(mesh_f, mesh_tuple, comments=None):
    vs, vt, vn, t_vs, t_vt, t_vn = mesh_tuple

    if os.path.splitext(mesh_f)[-1] == ".ply":
        from plyfile import PlyData, PlyElement
        from numpy.lib import recfunctions as rfn

        v_type  = [("x", "f4"), ("y", "f4"), ("z", "f4")]
        vi_type = [("vertex_indices", "i4", (3,))]
        tc_type = [("texcoord", "f4", (6,))]

        assert np.max(t_vs) < len(vs), f"got index {np.max(t_vs)}, #ver={len(vs)}"

        ver = rfn.unstructured_to_structured(vs, np.dtype(v_type))
        tri = rfn.unstructured_to_structured(t_vs, np.dtype(vi_type))

        if vt is not None and t_vt is not None:
            tex_coord = vt[t_vt.flatten()].reshape(-1, 6)

            tri = np.zeros((len(t_vs)), dtype=np.dtype(vi_type + tc_type))

            tri["vertex_indices"][:] = t_vs
            tri["texcoord"][:] = tex_coord

            assert np.max(tri["vertex_indices"]) < len(vs), f"got index {np.max(tri['vertex_indices'])}, #ver={len(vs)}"
        
        ver_el = PlyElement.describe(ver, 'vertex')
        tri_el = PlyElement.describe(tri, 'face')

        el_list = [ver_el, tri_el]

        comments = comments if comments is not None else []

        PlyData([ver_el, tri_el], comments=comments, text=True).write(mesh_f)