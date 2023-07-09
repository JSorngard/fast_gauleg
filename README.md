****************************************************************
Original C++ code: Copyright (C) 2014 by Ignace Bogaert  
This Rust implementation was written by Johanna Sörngård in 2023 
****************************************************************

This software package is based on the paper
   [I. Bogaert, "Iteration-Free Computation of Gauss-Legendre Quadrature Nodes and Weights"](https://doi.org/10.1137/140954969),
   published in volume 36, issue 3 of the SIAM Journal of Scientific Computing in 2014 on pages A1008 - A1026.

The main features of this software are:
- Speed: due to the simple formulas and the O(1) complexity computation of individual Gauss-Legendre 
  quadrature nodes and weights. This makes it compatible with parallel computing paradigms.
- Accuracy: the error on the nodes and weights is within a few ulps (see the paper for details).

Disclaimer:
THIS SOFTWARE IS PROVIDED "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR 
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.