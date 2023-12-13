// LICENSE
// =======
//
// Copyright (c) 2015, MILA (Montreal Institute for Learning Algorithms)
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//     * Neither the name of Theano nor the names of its contributors may be
//       used to endorse or promote products derived from this software without
//       specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ''AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
use std::collections::VecDeque;

use rand::Rng;

use crate::tests::generators::create_rng;

// https://github.com/mila-iqia/summerschool2015/blob/master/rnn_tutorial/synthetic.py
///
///     mackey_glass(sample_len=1000, tau=17, seed = None, n_samples = 1) -> input
///     Generate the Mackey Glass time-series. Parameters are:
///         - sample_len: length of the time-series in timesteps. Default is 1000.
///         - tau: delay of the MG - system. Commonly used values are tau=17 (mild
///           chaos) and tau=30 (moderate chaos). Default is 17.
///         - seed: to seed the random generator, can be used to generate the same
///           timeseries at each invocation.
///         - n_samples : number of samples to generate
pub fn mackey_glass(sample_len: usize, tau: usize, seed: Option<u64>) -> Vec<f64> {
    let mut delta_t = 10;
    let history_len = tau * delta_t;
    let mut timeseries = 1.2;

    let mut rng = create_rng(seed).unwrap();

    let mut history = VecDeque::with_capacity(history_len);
    for _i in 0..history_len {
        let val = 1.2 + 0.2 * (rng.gen::<f64>() - 0.5);
        history.push_back(val);
    }

    // Preallocate the array for the time-series
    let mut inp = vec![0.0; sample_len];

    for item in inp.iter_mut().take(sample_len) {
        for _ in 0..delta_t {
            let x_tau = history.pop_front().unwrap();
            history.push_back(timeseries);
            let last_hist = history[history.len() - 1];
            timeseries = last_hist
                + (0.2 * x_tau / (1.0 + x_tau.powi(10)) - 0.1 * last_hist) / delta_t as f64;
        }
        *item = timeseries;
    }
    // Squash timeseries through tanh
    inp.iter_mut().for_each(|v| *v = v.tanh());

    inp
}
