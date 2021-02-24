#   Copyright 2018 SciNet (https://github.com/eth-nn-physics/nn_physical_concepts)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from CoolProp.CoolProp import PropsSI
import numpy as np
import cPickle
import gzip
import io

"""
Parameters:
N: n_examples.
n_o: n_observations.
n_q: n_questions.
p_interval: in Pa.
T_interval: in K.
species1: in CoolProp notation. Heavier species.
species2: in CoolProp notation. Lighter species.
fileName: self-explanatory.
"""
def generate_data(N, n_o, p_interval=[1.0,2.0], T_interval=[1.0,2.0], M1=0.03995, M2=0.004, fileName=None):
    
    def getRho(pr, Tr, Z):
        Ru = 8.314 # Universal gas constant
        p = pr*1.0e5
        T = Tr*300.0
        return (p/(Ru * T)) * 1.0/((Z/M1) + ((1.0-Z)/M2))

    # Number of variables
    l_o = 3 # p, T, rho
    l_q = 2 # p, T
    l_a = 1 # rho

    # Sample the parameter, the binary mixture fraction of species1
    Z_sample = np.random.rand(N)

    x_o = np.empty([N, l_o*n_o])
    x_q = np.empty([N, l_q*1])
    x_a = np.empty([N, l_a*1])
    for i in range(N):
        pp_o = (p_interval[1] - p_interval[0]) * np.random.rand(n_o) + p_interval[0]
        TT_o = (T_interval[1] - T_interval[0]) * np.random.rand(n_o) + T_interval[0]
        for j in range(n_o):
            x_o[i, 0*n_o + j] = pp_o[j]
            x_o[i, 1*n_o + j] = TT_o[j]
            x_o[i, 2*n_o + j] = getRho(pp_o[j], TT_o[j], Z_sample[i])
        pp_q = (p_interval[1] - p_interval[0]) * np.random.rand() + p_interval[0]
        TT_q = (T_interval[1] - T_interval[0]) * np.random.rand() + T_interval[0]
        x_q[i, 0] = pp_q
        x_q[i, 1] = TT_q
        x_a[i] = getRho(pp_q, TT_q, Z_sample[i])
    result = ([x_o, x_q, x_a], Z_sample, [])
    if fileName is not None:
        f = gzip.open(io.data_path + fileName + ".plk.gz", 'wb')
        cPickle.dump(result, f, protocol=2)
        f.close()
    return (result)
