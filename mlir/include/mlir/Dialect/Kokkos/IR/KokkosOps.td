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
include "mlir/IR/RegionKindInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ParallelCombiningOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

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

#endif // KOKKOS_OPS_H
