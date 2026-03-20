#include "hdf5_io.hpp"

// ---------------------------------------------------------------------------
// Scalar attributes
// ---------------------------------------------------------------------------

void h5_write_attr(H5::Group& g, const string& name, double v)
{
    H5::DataSpace sp(H5S_SCALAR);
    H5::Attribute a = g.createAttribute(name, H5::PredType::NATIVE_DOUBLE, sp);
    a.write(H5::PredType::NATIVE_DOUBLE, &v);
}

void h5_write_attr(H5::Group& g, const string& name, int v)
{
    H5::DataSpace sp(H5S_SCALAR);
    H5::Attribute a = g.createAttribute(name, H5::PredType::NATIVE_INT, sp);
    a.write(H5::PredType::NATIVE_INT, &v);
}

void h5_write_attr(H5::Group& g, const string& name, const string& v)
{
    H5::StrType st(H5::PredType::C_S1, H5T_VARIABLE);
    H5::DataSpace sp(H5S_SCALAR);
    H5::Attribute a = g.createAttribute(name, st, sp);
    const char* cp = v.c_str();
    a.write(st, &cp);
}

double h5_read_attr_double(const H5::Group& g, const string& name)
{
    double v;
    H5::Attribute a = g.openAttribute(name);
    a.read(H5::PredType::NATIVE_DOUBLE, &v);
    return v;
}

int h5_read_attr_int(const H5::Group& g, const string& name)
{
    int v;
    H5::Attribute a = g.openAttribute(name);
    a.read(H5::PredType::NATIVE_INT, &v);
    return v;
}

string h5_read_attr_string(const H5::Group& g, const string& name)
{
    string v;
    H5::Attribute a = g.openAttribute(name);
    H5::StrType st = a.getStrType();
    a.read(st, v);
    return v;
}

// ---------------------------------------------------------------------------
// vector<double>
// ---------------------------------------------------------------------------

void h5_write_vec(H5::Group& g, const string& name, const vector<double>& v)
{
    hsize_t dims[1] = { v.size() };
    H5::DataSpace sp(1, dims);
    H5::DataSet ds = g.createDataSet(name, H5::PredType::NATIVE_DOUBLE, sp);
    if(!v.empty()) ds.write(v.data(), H5::PredType::NATIVE_DOUBLE);
}

void h5_read_vec(const H5::Group& g, const string& name, vector<double>& v)
{
    H5::DataSet ds = g.openDataSet(name);
    H5::DataSpace sp = ds.getSpace();
    hsize_t dims[1];
    sp.getSimpleExtentDims(dims);
    v.resize(dims[0]);
    if(dims[0] > 0) ds.read(v.data(), H5::PredType::NATIVE_DOUBLE);
}

// ---------------------------------------------------------------------------
// matrix<double>
// Stored as a sub-group containing:
//   attribute "r" : int
//   attribute "c" : int
//   dataset  "data" : 1-D double array (column-major flat layout)
// ---------------------------------------------------------------------------

void h5_write_mat(H5::Group& g, const string& name, const matrix<double>& m)
{
    H5::Group mg = g.createGroup(name);
    h5_write_attr(mg, "r", (int)m.r);
    h5_write_attr(mg, "c", (int)m.c);
    h5_write_vec(mg, "data", m.v);
}

void h5_read_mat(const H5::Group& g, const string& name, matrix<double>& m)
{
    H5::Group mg = g.openGroup(name);
    int r = h5_read_attr_int(mg, "r");
    int c = h5_read_attr_int(mg, "c");
    m.set_size((size_t)r, (size_t)c);
    h5_read_vec(mg, "data", m.v);
}

// ---------------------------------------------------------------------------
// matrix<Complex>
// Stored as a sub-group containing:
//   attribute "r" : int
//   attribute "c" : int
//   dataset  "re" : 1-D double array (real parts, column-major)
//   dataset  "im" : 1-D double array (imaginary parts, column-major)
// ---------------------------------------------------------------------------

void h5_write_mat(H5::Group& g, const string& name, const matrix<Complex>& m)
{
    H5::Group mg = g.createGroup(name);
    h5_write_attr(mg, "r", (int)m.r);
    h5_write_attr(mg, "c", (int)m.c);
    size_t n = m.r * m.c;
    vector<double> re(n), im(n);
    for(size_t k = 0; k < n; ++k) {
        re[k] = m.v[k].real();
        im[k] = m.v[k].imag();
    }
    h5_write_vec(mg, "re", re);
    h5_write_vec(mg, "im", im);
}

void h5_read_mat(const H5::Group& g, const string& name, matrix<Complex>& m)
{
    H5::Group mg = g.openGroup(name);
    int r = h5_read_attr_int(mg, "r");
    int c = h5_read_attr_int(mg, "c");
    m.set_size((size_t)r, (size_t)c);
    vector<double> re, im;
    h5_read_vec(mg, "re", re);
    h5_read_vec(mg, "im", im);
    for(size_t k = 0; k < (size_t)r * c; ++k)
        m.v[k] = Complex(re[k], im[k]);
}
