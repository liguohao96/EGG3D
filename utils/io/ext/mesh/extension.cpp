#include <vector>
#include <stdio.h>
#include <fstream>
#include <pybind11/numpy.h>

#include "extension.h"

namespace cpp{
    pybind11::dict read_ply_file(std::string& v){
		return pybind11::dict();
    }
    bool save_ply_file(std::string& filepath, pybind11::dict mesh_dict, bool is_binary){
		bool succ = true;

		using i32 = int32_t;
		using f32 = float;
		using u8  = uint8_t;

		using i32_arr = pybind11::array_t<i32>;
		using f32_arr = pybind11::array_t<f32>;
		using u8_arr  = pybind11::array_t<u8>;

		f32_arr gp;   // geometry vertex
		i32_arr gf;   // geometry face
		f32_arr uv;
		u8_arr  tc;   // texture color

		pybind11::object None = pybind11::none();
		pybind11::object gp_o = mesh_dict.attr("get")("geometry_vertex") ;
		pybind11::object gf_o = mesh_dict.attr("get")("geometry_face", None) ;
		pybind11::object uv_o = mesh_dict.attr("get")("texture_coord", None) ;
		pybind11::object tc_o = mesh_dict.attr("get")("texture_color", None) ;
		// pybind11::object vc_o = mesh_dict.attr("get")("geometry_face", None) ;
		pybind11::object tf_o = mesh_dict.attr("get")("texture_file",  None) ;

		gp = static_cast<f32_arr>(gp_o);
		auto gp_access = gp.unchecked<2>();

		std::vector<Comment> comments;
		std::vector<Element> elements;

		/*--------------comment------------------*/
		if (!tf_o.is(None)){
			std::string tf(static_cast<pybind11::str>(tf_o));
			comments.push_back(std::string("TextureFile ")+tf);
		}

		std::vector<std::shared_ptr<PropertyBase>> vertex_props;
		std::vector<std::shared_ptr<PropertyBase>> face_props;
		/*-----------element vertex---------------*/
		unsigned int num_ver  = gp_access.shape(0);
		auto vp_x_p = std::make_shared<TypeProperty<float>>("x", [&](unsigned int index, OutputAdapter& out){
			out << gp_access(index, 0);
		});
		auto vp_y_p = std::make_shared<TypeProperty<float>>("y", [&](unsigned int index, OutputAdapter& out){
			out << gp_access(index, 1);
		});
		auto vp_z_p = std::make_shared<TypeProperty<float>>("z", [&](unsigned int index, OutputAdapter& out){
			out << gp_access(index, 2);
		});

		vertex_props.push_back(vp_x_p);
		vertex_props.push_back(vp_y_p);
		vertex_props.push_back(vp_z_p);
		if (!tc_o.is(None)){
			auto tc_access = static_cast<u8_arr>(tc_o).unchecked<2>();
			if (tc_access.shape(0) == num_ver){
				using _type = unsigned char;
				auto r_p = std::make_shared<TypeProperty<_type>>("red", [tc_access](unsigned int index, OutputAdapter& out){
					out << tc_access(index, 0);
					// out << float(0);
				});
				auto g_p = std::make_shared<TypeProperty<_type>>("green", [tc_access](unsigned int index, OutputAdapter& out){
					out << tc_access(index, 1);
					// out << float(0);
				});
				auto b_p = std::make_shared<TypeProperty<_type>>("blue", [tc_access](unsigned int index, OutputAdapter& out){
					out << tc_access(index, 2);
					// out << float(0);
				});
				vertex_props.push_back(r_p);
				vertex_props.push_back(g_p);
				vertex_props.push_back(b_p);
			}
		}
		if (!uv_o.is(None)){
			auto uv_access = static_cast<f32_arr>(uv_o).unchecked<2>();
			if (uv_access.shape(0) == num_ver){
				auto u_p = std::make_shared<TypeProperty<float>>("texture_u", [uv_access](unsigned int index, OutputAdapter& out){
					out << uv_access(index, 0);
					// out << float(0);
				});
				auto v_p = std::make_shared<TypeProperty<float>>("texture_v", [uv_access](unsigned int index, OutputAdapter& out){
					out << uv_access(index, 1);
					// out << float(0);
				});
				vertex_props.push_back(u_p);
				vertex_props.push_back(v_p);
			}
		}
		elements.push_back(Element("vertex", num_ver, vertex_props));

		if (!gf_o.is(None)){
			/*------------element face----------------*/
			auto gf_access = static_cast<i32_arr>(gf_o).unchecked<2>();
			unsigned int num_face = gf_access.shape(0);

			auto gf_vi_p = std::make_shared<ListProperty<int>>("vertex_indices", [&](unsigned int index, OutputAdapter& out){
                if (out.is_binary){
                    out << static_cast<char>(3) \
                        << gf_access(index, 0)  \
                        << gf_access(index, 1)  \
                        << gf_access(index, 2);
                }else{
                    out << static_cast<char>(3) << " " \
                        << gf_access(index, 0) << " " \
                        << gf_access(index, 1) << " " \
                        << gf_access(index, 2);
                }
			});

			face_props.push_back(gf_vi_p);

			if (!uv_o.is(None)){
				auto uv_access = static_cast<f32_arr>(uv_o).unchecked<2>();
				if (uv_access.shape(0) == num_face){
					auto uv_p = std::make_shared<ListProperty<float>>("texcoord", [uv_access](unsigned int index, OutputAdapter& out){
						if (out.is_binary){
							out << static_cast<char>(6) \
								<< uv_access(index, 0)  \
								<< uv_access(index, 1)  \
								<< uv_access(index, 2)  \
								<< uv_access(index, 3)  \
								<< uv_access(index, 4)  \
								<< uv_access(index, 5);
						}else{
							out << static_cast<char>(6) << " " \
								<< uv_access(index, 0) << " " \
								<< uv_access(index, 1) << " " \
								<< uv_access(index, 2) << " " \
								<< uv_access(index, 3) << " " \
								<< uv_access(index, 4) << " " \
								<< uv_access(index, 5);
						}
					});
					face_props.push_back(uv_p);
				}
			}

			elements.push_back(Element("face", num_face, face_props));
		}


		// printf("%d\n", std::is_same<unsigned char, uint8_t>::value);
		PLYWriter writer(comments, elements);
		succ = writer.save(filepath, is_binary);

		return succ;
	}
} // namespace cpp