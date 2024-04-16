<table>
  <tr>
    <td>
      <img src="lse_logo.png" alt="overview" width="200"/>
    </td>
    <td>
      <h1>Variational-LSE-Solver</h1>
    </td>
  </tr>
</table>

[![Static Badge](https://img.shields.io/badge/arXiv-2404.09916-red)
](https://arxiv.org/abs/2404.09916)[![Static Badge](https://img.shields.io/badge/PyPI-pip_install_variational--lse--solver-blue)](https://pypi.org/project/variational-lse-solver/)

This repo contains the code for the PennyLane-based `variational-lse-solver` library introduced in 
["Comprehensive Library of Variational LSE Solvers", N. Meyer et al. (2024)](https://arxiv.org/abs/2404.09916).

> Linear systems of equations can be found in various mathematical domains, as well as in the field of machine learning. By employing noisy intermediate-scale quantum devices, variational solvers promise to accelerate finding solutions for large systems. Although there is a wealth of theoretical research on these algorithms, only fragmentary implementations exist. To fill this gap, **we have developed the variational-lse-solver framework**, which **realizes existing approaches in literature**, and **introduces several enhancements**. The user-friendly interface is designed for researchers that work at the **abstraction level of identifying and developing end-to-end applications**.

## Setup and Installation

The library requires an installation of `python 3.12`, and following libraries:
- `pennylane~=0.34`
- `torch~=2.1.2`
- `tqdm~=4.66.2`

We recommend setting up a conda environment:

```
conda create --name ENV_NAME python=3.12
conda activate ENV_NAME
```

The package `variational-lse-solver` can be installed locally via:
```
cd variational-lse-solver
pip install -e .
```

## Usage and Reproduction of Results

Information on how to use the different modalities of the libraries are described in the documentation.
Additionally, we provide to usage examples.

#### Reproduce Paper Results

To reproduce the results depicted in the paper, one can run

```
python examples/reproduce_result.py --mode 'pauli' --method 'hadamard'
```
Keep in mind, that the reported results were averaged over 100 random initializations. 
In order to use the local cost function, just add the `--local` flag. 
It is also possible to use the matrix decomposition modes `unitary` and `circuit`, and the loss evaluation methods `overlap` and `coherent`.
Running the script will print the solution found with the variational LSE solver, as well as the classically validated solution.

#### Solve Random LSE

To solve a randomly generated LSE with a system matrix of size `8 x 8`, one can run
```
python examples/test_random_system.py --threshold 5e-5
```
In order to use the local cost function, just add the `--local` flag. 
To increase the accuracy of the solution, reduce the value of `--threshold`.
The implementation by default uses the `direct` mode, i.e. no matrix decomposition is performed.
Running the script wil print the solution found with the variational LSE solver, as well as the classically validated solution.

## Acknowledgements

The `variational-lse-solver` library is mostly based on the techniques introduced in
["Variational Quantum Linear Solver", C. Bravo-Prieto et al., Quantum 7, 1188 (2023)](https://quantum-journal.org/papers/q-2023-11-22-1188/).

The concept of using dynamically growing circuits is inspired by
["Variational quantum linear solver with a dynamic ansatz", H. Patil et al., Phys. Rev. A 105, 012423 (2022)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.105.012423)

There are some alternative implementations of subroutines provided by `variational-lse-solver` in the PennyLane Demos:
- https://pennylane.ai/qml/demos/tutorial_vqls/
- https://pennylane.ai/qml/demos/tutorial_coherent_vqls/

However, those realisations contain several hard-coded parts and can not be used for arbitrary problems out of the box.
Furthermore, we identified some small inaccuracies, which might lead to converging to a wrong solution 
-- this is explained in more detail in the documentation of `variational-lse-solver`.

## Citation

If you use the `variational-lse-solve` library or results from the paper, please cite our work as

```
@article{meyer2024comprehensive,
  title={Comprehensive Library of Variational LSE Solvers},
  author={Meyer, Nico and R"\ohn, Martin and Murauer, Jakob and Scherer, Daniel D. and Plinge, Axel and Mutschler, Christopher},
  journal={arXiv:2404.09916},
  year={2024},
  doi={.../arXiv...}
}
```

## Version History

Initial release (v1.0): April 2024

## License

Apache 2.0 License
  