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

def ParallelUnitStep : Pass<"parallel-unit-step", "ModuleOp"> {
  let summary = "Rewrites scf.parallel ops to have constant, unit step size in all dimensions";
  let description = [{
  }];
  let constructor = "mlir::createKokkosLoopMappingPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "scf::SCFDialect"
  ];
}

def KokkosLoopMapping : Pass<"kokkos-loop-mapping", "ModuleOp"> {
  let summary = "Maps nested SCF parallel loops to Kokkos parallel loops";
  let description = [{
  }];
  let constructor = "mlir::createKokkosLoopMappingPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "bufferization::BufferizationDialect",
    "kokkos::KokkosDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

def KokkosMemorySpaceAssignment : Pass<"kokkos-assign-memory-spaces", "ModuleOp"> {
  let summary = "Determines the MemorySpace attributes for each memref value";
  let description = [{
  }];
  let constructor = "mlir::createKokkosMemorySpaceAssignmentPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "bufferization::BufferizationDialect",
    "kokkos::KokkosDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

def KokkosDualViewManagement : Pass<"kokkos-dualview-management", "ModuleOp"> {
  let summary = "Insert sync and modify operations for DualView-typed memrefs";
  let description = [{
  }];
  let constructor = "mlir::createKokkosDualViewManagementPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "bufferization::BufferizationDialect",
    "kokkos::KokkosDialect",
    "linalg::LinalgDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

#endif // MLIR_DIALECT_KOKKOS_TRANSFORMS_PASSES
