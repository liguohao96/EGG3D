#ifndef __EXTENSION_H__
#define __EXTENSION_H__

#include <vector>
#include <functional>
#include <fstream>
#include <sstream>
#include <cstdint>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

inline bool is_little_endian(){
    int32_t a = 0x01;
    char*   b = (char*)(&a);
    return (b[0]==0x01);
};
namespace cpp{
    template<typename T>
    const std::string ply_typename(){
        if (std::is_same<T, int32_t>::value)
            return "int";
        else if (std::is_same<T, float>::value)
            return "float";
        else if (std::is_same<T, double>::value)
            return "double";
        else if (std::is_same<T, unsigned char>::value)
            return "uchar";
        else
            return "int";
    };
    class OutputAdapter{
    public:
        OutputAdapter(std::ostream& ostream, bool is_bin)
        :_ostream(ostream), is_binary(is_bin)
        {}
        template<typename T>
        OutputAdapter& operator<<(const T& obj){
            if (is_binary){
                _ostream.write(reinterpret_cast<const char*>(&obj), sizeof(T));
            }else{
                if (sizeof(T) == 1)
                    _ostream << (unsigned long long)(obj);
                else
                    _ostream << obj;
            }
            return (*this);
        }
        void putchar(const char c){
            _ostream << c;
        }
        const bool    is_binary;
    protected:
        std::ostream& _ostream;
    };

    using W_FUNC = std::function<void(unsigned int, OutputAdapter&)>;
    using R_FUNC = std::function<void(unsigned int, std::istream&)>;

    class PropertyBase{
    public:
        PropertyBase(const std::string& name, const std::string& type, const W_FUNC& w_hook)
        :_name(name), _type(type), _write_hook(w_hook){
        }
        const std::string name() const {
            return _name;
        }
        virtual const std::string meta_info() const = 0;
        // virtual const std::string meta_info() const {
        //     std::stringstream ss;
        //     ss << "property " << PropertyBase::_type << " " << PropertyBase::_name;
        //     return ss.str();
        // }
        // virtual bool set_write_hook(const std::function<void(unsigned int, std::ostream&)>&& w_hook){
        //     _write_hook = w_hook;
        // }
        virtual void write_data_at(unsigned int index, OutputAdapter& out) const{
            _write_hook(index, out);
        }
        virtual ~PropertyBase(){
        }

    protected:
        std::string _name;
        std::string _type;
        W_FUNC      _write_hook;
    };
    template<typename T>
    class TypeProperty: public PropertyBase{
    public:
        TypeProperty(const std::string& name, const W_FUNC& w_hook)
        :PropertyBase(name, ply_typename<T>(), w_hook)
        {
        }
        virtual const std::string meta_info() const override {
            std::stringstream ss;
            ss << "property " << PropertyBase::_type << " " << PropertyBase::_name;
            return ss.str();
        }
        virtual ~TypeProperty(){}
    };
    template<typename T>
    class ListProperty: public TypeProperty<T>{
    public:
        ListProperty(const std::string& name, const W_FUNC& w_hook)
        :TypeProperty<T>(name, w_hook)
        {
        }
        virtual const std::string meta_info() const override {
            std::stringstream ss;
            ss << "property list uchar " << TypeProperty<T>::_type << " " << TypeProperty<T>::_name;
            return ss.str();
        }
        virtual ~ListProperty(){}
    };
    using Comment = std::string;

    class Element{
    public:
        Element(const std::string& name, unsigned int size, std::vector<std::shared_ptr<PropertyBase>>& propertys)
        :_name(name), _size(size), _propertys(propertys)
        {}
        const std::string meta_string() const{
            std::stringstream ss;
            ss << "element " << _name << " " << _size << std::endl;
            for (auto& p : _propertys)
                ss << p->meta_info() << std::endl;
            return ss.str();
        }
        void write_data(OutputAdapter& out) const{
            unsigned int num_p = _propertys.size();
            for(unsigned int i=0; i<_size; ++i){
                for(unsigned int j=0; j<num_p; ++j){
                    _propertys[j]->write_data_at(i, out);
                    if (!out.is_binary){
                        if (j<num_p-1)
                            out.putchar(' ');
                        else
                            out.putchar('\n');
                    }
                }
            }
        }
    private:
        std::string                                 _name;
        unsigned int                                _size;
        std::vector<std::shared_ptr<PropertyBase>>  _propertys;
    };
    class PLYWriter{
    public:
        PLYWriter(const std::vector<Comment>& comments, const std::vector<Element>& elements)
        :_comments(comments),_elements(elements){
        }

        bool save(const std::string& filepath, const bool is_binary) const {
            std::ofstream fout;
            if (is_binary){
                fout.open(filepath, std::ios::out|std::ios::binary);
            }else{
                fout.open(filepath, std::ios::out);
            }
            if (!fout.is_open())
                return false;

            fout << "ply"       << std::endl;
            if (is_binary)
                if (is_little_endian())
                    fout << "format binary_little_endian 1.0" << std::endl;
                else
                    fout << "format binary_big_endian 1.0" << std::endl;
            else
                fout << "format ascii 1.0" << std::endl;

            for(auto& c : _comments)
                fout << "comment " << c << std::endl;

            for(auto& e : _elements)
                fout << e.meta_string();
            fout << "end_header\n";

            OutputAdapter output_adapter{fout, is_binary};

            for(auto& e : _elements)
                e.write_data(output_adapter);

            fout.close();

            return true;
        }

    private:
        std::string          _filepath;
        bool                 _is_binary;
        std::vector<Comment> _comments;
        std::vector<Element> _elements;

    };

    pybind11::dict load_ply_file(std::string& filename);
    bool           save_ply_file(std::string& filename, pybind11::dict mesh_dict, bool is_binary);
} // namespace cpp

#endif