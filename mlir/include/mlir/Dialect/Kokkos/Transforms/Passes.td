//===-- Passes.td - Kokkos pass definition file -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_KOKKOS_TRANSFORMS_PASSES
#define MLIR_DIALECT_KOKKOS_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def SparseKokkosCodegen : Pass<"sparse-kokkos-codegen", "ModuleOp"> {
  let summary = "Generates Kokkos code during sparsification";
  let description = [{
    Enables sparse compiler to generate Kokkos code, which can use non-host backends.
  }];
  let constructor = "mlir::createSparseKokkosCodegenPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "bufferization::BufferizationDialect",
    "emitc::EmitCDialect",
    "kokkos::KokkosDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}


#endif // MLIR_DIALECT_KOKKOS_TRANSFORMS_PASSES
