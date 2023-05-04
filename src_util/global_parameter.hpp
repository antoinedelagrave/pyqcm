#ifndef global_parameter_h
#define global_parameter_h

#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <cstdarg>
#include <cstdio>
#include <assert.h>

#include "parser.hpp"

using namespace std;

enum H_FORMAT {H_format_csr, H_format_ops, H_format_onthefly, H_format_factorized, H_format_dense, H_format_eigen};
extern H_FORMAT Hamiltonian_format;

//! scheme for handling global parameters and options
/**
the template type is meant to be bool, int, double or string
*/
template<typename T>
struct global_parameter{
  string name; //!< name of the parameter
  string description; //!< description, for help
  T value; //!< current value of the parameter
  T default_value; //!< default value of the parameter
  
  global_parameter() : value(T()), default_value(T()), name(string()), description(string()) {}
  
  global_parameter(T _def, const string &_name, const string &_des) : value(_def), default_value(_def), name(_name), description(_des) {}
  
  /**
   Prints the description/help to a latex file
   */
  void print_latex(ostream &out){
    out << "\\cd{" << name << "}\n&" << std::boolalpha << value << "\n&"  << description << "\\\\\n";
  }
  
  /**
   Prints the description/help to a RST file, such as the ones used in Sphinx documentation
   */
  void print_RST(ostream &out){
    out << "    \"" << name << "\", \"" << std::boolalpha << default_value << "\", \""  << description << "\"" << endl;
  }
};

template<typename T>
ostream & operator<<(ostream &s, global_parameter<T> &g){
  s << g.name << " : "  << g.description << "  [default = " << g.default_value << ']';
  return s;
}


void Print_global_parameters_latex();
void Print_global_parameters_RST();
void Print_global_parameters(ostream &out);

bool global_bool(const string& name);
size_t global_int(const string& name);
double global_double(const string& name);
char global_char(const string& name);
void new_global_bool(bool def, const string& name, const string& des);
void new_global_int(size_t def, const string& name, const string& des);
void new_global_double(double def, const string& name, const string& des);
void new_global_char(char def, const string& name, const string& des);
void set_global_bool(const string& param, bool value);
void set_global_double(const string& param, double value);
void set_global_int(const string& param, size_t value);
void set_global_char(const string& param, char value);
void print_options(int to_file = 0);
bool is_global_bool(const string& name);
bool is_global_int(const string& name);
bool is_global_double(const string& name);
bool is_global_char(const string& name);


#endif
