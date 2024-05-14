//===- KokkosPasses.cpp - Passes for lowering to Kokkos dialect -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Kokkos/IR/Kokkos.h"
#include "mlir/Dialect/Kokkos/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_SPARSEKOKKOSCODEGEN
#include "mlir/Dialect/Kokkos/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::kokkos;

namespace {

struct SparseKokkosCodegenPass
    : public impl::SparseKokkosCodegenBase<SparseKokkosCodegenPass> {

  SparseKokkosCodegenPass() = default;
  SparseKokkosCodegenPass(const SparseKokkosCodegenPass& pass) = default;

  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    populateSparseKokkosCodegenPatterns(patterns);
    (void) applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
  }
};
}

std::unique_ptr<Pass> mlir::createSparseKokkosCodegenPass() {
  return std::make_unique<SparseKokkosCodegenPass>();
}

