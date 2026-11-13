# Learning Advanced Differential Privacy Mechanisms with Fixed Data Order

Implementation of advanced differential privacy mechanisms developed during my Semester and Master's thesis at ETH Zurich.
The thesis provides the mathematical background, theoretical analysis, algorithm descriptions, and implementation details.
The complete thesis is publicly available:

[Read the Master's thesis (PDF)](https://www.its.uni-luebeck.de/fileadmin/files/theses/MA_NadjaAoutouf_LearnDPMechanisms.pdf)

## Overview

Differential privacy provides a mathematical framework for protecting individual information when analyzing sensitive datasets. However, designing differentially private mechanisms that achieve high accuracy can be challenging, particularly when the underlying problem has a complex structure.
This work investigates how **problem-dependent differentially private mechanisms can be learned from publicly available data**, with the goal of improving their accuracy while maintaining privacy guarantees.
The work builds on a framework for learning differentially private approximations and studies several extensions to improve its applicability to more general problems.

## Main Contributions

The thesis explores the following aspects of learning differentially private mechanisms:

- **Sparse datasets:** Investigating how the framework can be adapted to handle sparse data.
- **Optimal abort conditions:** Studying conditions under which the mechanism should terminate without producing an output.
- **Multivariate inputs:** Extending the framework to problems involving multiple input variables.
- **Smoothed sensitivity:** Incorporating sensitivity-based techniques to improve the accuracy of the mechanism.
- **Decision trees:** Applying the framework to decision-tree models as a concrete problem instantiation.

The implementation includes algorithms for initializing the problem structure, evaluating candidate solutions, finding suitable center values, and determining valid output ranges.

## Implementation

The repository contains the Python implementation presented in **Appendix A** of the Master's thesis.

The code is organized around the following components:

| Component | Description |
|---|---|
| Problem Instantiations | Definition of the decision-tree model |
| Initialization | Initialization of the problem structure and assignment of nodes to ranks |
| Helper Functions | Functions for evaluation, updating buffers, and synchronization |
| Finding Suitable Center Values | Algorithm for identifying suitable center values |
| Finding Output Ranges | Algorithm for determining valid output ranges |

The implementation is preserved as presented in the original 2021 thesis, without algorithmic modifications or modernization.
