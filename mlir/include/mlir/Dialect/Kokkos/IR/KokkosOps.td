//===- KokkosOps.td - Kokkos dialect ops ------*- tablegen -*-===//

// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

#ifndef KOKKOS_OPS_H
#define KOKKOS_OPS_H

include "mlir/Dialect/Kokkos/IR/KokkosBase.td"
include "mlir/Dialect/MemRef/IR/MemRefBase.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ParallelCombiningOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

// Base class.
class Kokkos_Op<string mnemonic, list<Trait> traits = []>
    : Op<Kokkos_Dialect, mnemonic, traits>;

def ParallelOp : Kokkos_Op<"parallel",
    [AutomaticAllocationScope,
     AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<LoopLikeOpInterface>,
     RecursiveMemoryEffects,
     DeclareOpInterfaceMethods<RegionBranchOpInterface>,
     SingleBlockImplicitTerminator<"scf::YieldOp">]> {
  let summary = "Kokkos base parallel for operation";
  let description = [{
  }];

  let arguments = (ins Kokkos_ExecutionSpace:$executionSpace,
                       Kokkos_ParallelLevel:$parallelLevel,
                       Variadic<Index>:$lowerBound,
                       Variadic<Index>:$upperBound,
                       Variadic<AnyType>:$initVals);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::mlir::kokkos::ExecutionSpace":$executionSpace, "::mlir::kokkos::ParallelLevel":$parallelLevel, "ValueRange":$lowerBounds, "ValueRange":$upperBounds,
      "ValueRange":$initVals,
      CArg<"function_ref<void (OpBuilder &, Location, ValueRange, ValueRange)>",
           "nullptr">:$bodyBuilderFn)>,
    OpBuilder<(ins "::mlir::kokkos::ExecutionSpace":$executionSpace, "::mlir::kokkos::ParallelLevel":$parallelLevel, "ValueRange":$lowerBounds, "ValueRange":$upperBounds,
      CArg<"function_ref<void (OpBuilder &, Location, ValueRange)>",
           "nullptr">:$bodyBuilderFn)>,
  ];

  let extraClassDeclaration = [{
    ValueRange getInductionVars() {
      return getBody()->getArguments();
    }
    unsigned getNumLoops() { return getLowerBound().size(); }
    unsigned getNumReductions() { return getInitVals().size(); }
  }];

  let hasCanonicalizer = 1;
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def SyncOp : Kokkos_Op<"sync"> {

  let description = [{
    For a memref with DualView memory space, sync its contents to the
    provided memory space if it has been marked modified in the opposite space.
  }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "the memref to sync">:$view,
                       Kokkos_MemorySpace:$memorySpace);

  let assemblyFormat = [{
    $view attr-dict `:` type($view)
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

def ModifyOp : Kokkos_Op<"modify"> {
  let description = [{
    For a memref with DualView memory space, mark its contents as modified
    in the given space. This operation is idempotent.
  }];

  let arguments = (ins Arg<AnyRankedOrUnrankedMemRef, "the memref to mark modified">:$view,
                       Kokkos_MemorySpace:$memorySpace);

  let assemblyFormat = [{
    $view attr-dict `:` type($view)
  }];

  let hasCanonicalizer = 1;
  let hasFolder = 1;
}

// TODO: allow SingleOp to have results (broadcast mode)
// - Add InferTypeOpAdaptor trait
// - Add: let results = (outs Variadic<AnyType>:$results);

def SingleOp : Kokkos_Op<"single", [SingleBlockImplicitTerminator<"scf::YieldOp">, RecursiveMemoryEffects, NoRegionArguments]> {
  let summary = "Restricts a block to execute exactly once per team or per thread";
  let description = [{
    "Restricts a block to execute exactly once per team or per thread"
  }];

  let arguments = (ins Kokkos_SingleLevel:$level);
  let regions = (region SizedRegion<1>:$region);

  let hasFolder = 0;
  let hasCanonicalizer = 0;
  let hasCustomAssemblyFormat = 0;
  let hasVerifier = 0;
}

#endif // KOKKOS_OPS_H
