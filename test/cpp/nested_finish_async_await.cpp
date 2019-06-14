#include "hclib_cpp.h"
#include <iostream>
int main (int argc, char *argv[])
{
  int n_rounds = 100, n_tiles=5;
  const char *deps[] = { "system" };
  hclib::launch(deps, 1, [&]() {
      //create a 2D array of promises
      auto tile_array = new hclib::promise_t<void*>**[n_rounds+1];
      for (int tt=0; tt < n_rounds+1; ++tt) {
          tile_array[tt] = new hclib::promise_t<void*>*[n_tiles];
          for (int j=0; j < n_tiles; ++j) {
              tile_array[tt][j] = new hclib::promise_t<void*>();
          }
      }
      for (int j=0; j < n_tiles; ++j)
          tile_array[0][j]->put(nullptr);
      
      for (int tt=0; tt < n_rounds; ++tt){
          for (int j=0; j<n_tiles; ++j) {
              int left_nbr = (n_tiles+j-1)%n_tiles;
              int right_nbr = (j+1)%n_tiles;
              hclib::async_await([=]{
                  hclib::finish([=]{
                      hclib::async_await([=]{
                          tile_array[tt+1][j]->put(nullptr);                       
                      }, tile_array[tt][left_nbr]->get_future());                                 
                  });
              }, tile_array[tt][left_nbr]->get_future(),
              tile_array[tt][right_nbr]->get_future());
          }
          std::cout<<"finished "<< tt+1 <<" / "<<n_rounds<<std::endl;
      }
  });
  return 0;
}
