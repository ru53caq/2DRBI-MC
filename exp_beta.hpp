/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_TUTORIALS_MC_ISING2_EXP_BETA_HPP_9acbc85664694341b6ccec3bbda91e31
#define ALPS_TUTORIALS_MC_ISING2_EXP_BETA_HPP_9acbc85664694341b6ccec3bbda91e31

#include <cassert>
#include <cmath>
#include <vector>

/// Functor class for table-lookup implementation of exp(beta*delta) for 3D Ising Plaquette. Note: no "minus" sign!
class exp_beta {
  public:
    const static int TABLE_MIN_IDX=-24;
    const static int TABLE_MAX_IDX=24;
    const static unsigned int TABLE_SIZE=TABLE_MAX_IDX-TABLE_MIN_IDX+1;
  private:
    std::vector<double> table_;
    double beta_;

    /// Converts integer argument to index in the table
    static unsigned int arg2idx_(int arg) {
        assert(-24==arg || -22==arg || -20==arg || -18==arg || -16==arg || -14==arg || -12==arg ||-10==arg || -8==arg || -6==arg || -4==arg || -2==arg || 0==arg|| 2==arg || 4==arg || 6==arg || 8==arg || 10==arg || 12==arg || 14==arg || 16==arg || 18==arg || 20==arg || 22==arg || 24==arg);
        unsigned int idx=arg-TABLE_MIN_IDX;
        assert(idx<TABLE_SIZE);
        return idx;
    }

    /// Fills a table slot for the given argument
    void set_slot_(int arg) {
        unsigned int idx=arg2idx_(arg);
        table_[idx]=exp(beta_*arg);
    }

  public:

    /// Constructs the function object to compute exp(beta*x) with the given beta
    explicit exp_beta(double beta): table_(TABLE_SIZE,-1.0), beta_(beta) {
        for (int i=-12; i<13;++i){
        set_slot_(2*i);}
    }

    /// Evaluates exp(beta*x) for the given argument x
    double operator()(int x) const {
        double v=table_[arg2idx_(x)];
        assert(!(v<0.) && "Hit an empty slot");
        return v;
    }
};

#endif /* ALPS_TUTORIALS_MC_ISING2_EXP_BETA_HPP_9acbc85664694341b6ccec3bbda91e31 */
