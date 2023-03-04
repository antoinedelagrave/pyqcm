#include "global_parameter.hpp"
#include "lattice_model_instance.hpp"
#include "QCM.hpp"

/**
 This function computes the density and bond profiles for the model instance
 */
pair<vector<array<double,9>>, vector<array<complex<double>, 11>>> lattice_model_instance::site_and_bond_profile()
{
  const complex<double> Imag(0,1); // i
  vector<array<double, 9>> site_ave(model->sites.size());
  vector<array<complex<double>, 11>> bond_ave(model->bonds.size());
  
  if(!gf_solved) Green_function_solve();

  for(int i=0; i<site_ave.size(); i++){
    site_ave[i].fill(0);
    vector3D<double> R = model->phys.to(model->sites[i].position);
    site_ave[i][0] = R.x;
    site_ave[i][1] = R.y;
    site_ave[i][2] = R.z;
  }
  for(int i=0; i<bond_ave.size(); i++){
    bond_ave[i].fill(0);
    vector3D<double> R = model->phys.to(model->sites[model->bonds[i].first].position);
    vector3D<double> Ri = model->phys.to(model->sites[model->bonds[i].second].position);
    bond_ave[i][0] = complex<double>(R.x, Ri.x);
    bond_ave[i][1] = complex<double>(R.y, Ri.y);
    bond_ave[i][2] = complex<double>(R.z, Ri.z);
  }

  // loop over clusters
  int s_off = 3;
  int b_off = 3;
  for(int clus = 0; clus < model->clusters.size(); clus++){
    matrix<complex<double>> G_ave = ED::Green_function_average(false, clus);
    matrix<complex<double>> G_ave_down;
    if(model->mixing == HS_mixing::up_down) G_ave_down = ED::Green_function_average(true, clus);
    size_t dim = ED::Green_function_dimension((size_t)clus);
    auto n_sites = model->clusters[clus].n_sites;
    auto offset = model->clusters[clus].offset;
    for(int i=0; i<n_sites; i++){
      int id = i+(int)n_sites;
      int id3 = i+3*(int)n_sites;
      switch(model->mixing){
        case HS_mixing::normal:
          site_ave[i + offset][0+s_off] = 2*G_ave(i,i).real();
          break;
        case HS_mixing::anomalous:
          site_ave[i + offset][0+s_off] = G_ave(i,i).real() - G_ave(id, id).real() + 1.0;
          site_ave[i + offset][3+s_off] = G_ave(i,i).real() + G_ave(id, id).real() - 1.0;
          site_ave[i + offset][4+s_off] = 2.0*G_ave(i,id).real();
          site_ave[i + offset][5+s_off] = 2.0*G_ave(i,id).imag();
          break;
        case HS_mixing::spin_flip:
          site_ave[i + offset][0+s_off] = (G_ave(i,i) + G_ave(id, id)).real();
          site_ave[i + offset][1+s_off] = (G_ave(i,id) + G_ave(id, i)).real();
          site_ave[i + offset][2+s_off] = (G_ave(i,id) - G_ave(id, i)).imag();
          site_ave[i + offset][3+s_off] = (G_ave(i,i) - G_ave(id, id)).real();
          break;
        case HS_mixing::full:
          site_ave[i + offset][0+s_off] = (G_ave(i,i) + G_ave(id, id)).real();
          site_ave[i + offset][1+s_off] = (G_ave(i,id) + G_ave(id, i)).real();
          site_ave[i + offset][2+s_off] = (G_ave(i,id) - G_ave(id, i)).imag();
          site_ave[i + offset][3+s_off] = (G_ave(i,i) - G_ave(id, id)).real();
          site_ave[i + offset][4+s_off] = G_ave(i,id3).real();
          site_ave[i + offset][5+s_off] = G_ave(i,id3).imag();
          break;
        case HS_mixing::up_down:
          site_ave[i + offset][0+s_off] = (G_ave(i,i) + G_ave_down(i, i)).real();
          site_ave[i + offset][3+s_off] = (G_ave(i,i) - G_ave_down(i, i)).real();
          break;
      }
    }
    for(int i=0; i<bond_ave.size(); i++){
      size_t I = model->bonds[i].first;
      size_t J = model->bonds[i].second;
      size_t Ic = I - offset;
      size_t Jc = J - offset;
      size_t Icd = Ic + n_sites;
      size_t Jcd = Jc + n_sites;
      size_t Jcd2 = Jc + 2*n_sites;
      size_t Icd3 = Ic + 3*n_sites;
      size_t Jcd3 = Jc + 3*n_sites;
      if(model->sites[I].cluster != clus) continue;
      switch(model->mixing){
        case HS_mixing::normal:
          bond_ave[i][0+b_off] = 4.0*G_ave(Ic, Jc);
          break;
        case HS_mixing::anomalous:
          bond_ave[i][0+b_off] = 2.0*(G_ave(Ic, Jc)-G_ave(Jcd, Icd));
          bond_ave[i][4+b_off] = 2.0*(G_ave(Ic, Jcd) + G_ave(Jc, Icd));
          bond_ave[i][7+b_off] = 2.0*(G_ave(Ic, Jcd) - G_ave(Jc, Icd));
          break;
        case HS_mixing::spin_flip:
          bond_ave[i][0+b_off] = 2.0*(G_ave(Ic, Jc) + G_ave(Icd, Jcd));
          bond_ave[i][1+b_off] = 2.0*(G_ave(Ic, Jcd) + G_ave(Jc, Icd));
          bond_ave[i][2+b_off] = -2.0*Imag*(G_ave(Ic, Jcd) - G_ave(Jc, Icd));
          bond_ave[i][3+b_off] = 2.0*(G_ave(Ic, Jc)  - G_ave(Icd, Jcd));
          break;
        case HS_mixing::full:
          bond_ave[i][0+b_off] = 2.0*(G_ave(Ic, Jc)  + G_ave(Icd, Jcd));
          bond_ave[i][1+b_off] = 2.0*(G_ave(Ic, Jcd) + G_ave(Jc, Icd));
          bond_ave[i][2+b_off] = -2.0*Imag*(G_ave(Ic, Jcd) - G_ave(Jc, Icd));
          bond_ave[i][3+b_off] = 2.0*(G_ave(Ic, Jc)  - G_ave(Icd, Jcd));
          bond_ave[i][4+b_off] = 2.0*(G_ave(Ic, Jcd3) + G_ave(Jc, Icd3));
          bond_ave[i][5+b_off] = -2.0*(G_ave(Ic, Jcd2) + G_ave(Jcd, Icd3));
          bond_ave[i][6+b_off] = -2.0*Imag*(G_ave(Ic, Jcd2) - G_ave(Jcd, Icd3));
          bond_ave[i][7+b_off] = 2.0*(G_ave(Ic, Jcd3) - G_ave(Jc, Icd3));
          break;
        case HS_mixing::up_down:
          bond_ave[i][0+b_off] = 2.0*(G_ave(Ic, Jc) + G_ave_down(Ic, Jc));
          bond_ave[i][3+b_off] = 2.0*(G_ave(Ic, Jc) - G_ave_down(Ic, Jc));
          break;
      }
    }
  }
  return {site_ave, bond_ave};
}

