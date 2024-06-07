// RUN: mlir-opt %s -part-compiler
// RUN: mlir-opt %s -part-compiler

#SortedCOO = #sparse_tensor.encoding<{
  lvlTypes = [ "compressed-nu", "singleton" ]
}>
#partEncoding = #part_tensor.encoding<{
  partConst = 1,
  sparseAttributes = #SortedCOO
}>
module {
  func.func @dumpPartitions(%A: tensor<?x?xf32, #partEncoding>) -> memref<?xindex> {
    %partition_plan = part_tensor.get_partitions %A: tensor<?x?xf32, #partEncoding> -> memref<?xindex>
    return %partition_plan: memref<?xindex>
  }


  func.func @par1(%arg0: memref<?x?x?xf64>) {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c5 = arith.constant 5 : index
    scf.parallel (%arg5) = (%c0) to (%c5) step (%c1) {
      %0 = memref.load %arg0[%arg5] : memref<?xf64>
      %1 = memref.load %arg1[%arg5] : memref<?xindex>
      %2 = arith.addi %arg5, %c1 : index
      %3 = memref.load %arg1[%2] : memref<?xindex>
      %4 = scf.parallel (%arg6) = (%1) to (%3) step (%c1) init (%0) -> f64 {
        %5 = memref.load %arg2[%arg6] : memref<?xindex>
        %6 = memref.load %arg3[%arg6] : memref<?xf64>
        %7 = memref.load %arg4[%5] : memref<?xf64>
        %8 = arith.mulf %6, %7 : f64
        scf.reduce(%8)  : f64 {
        ^bb0(%arg7: f64, %arg8: f64):
          %9 = arith.addf %arg7, %arg8 : f64
          scf.reduce.return %9 : f64
        }
        scf.yield
      } {"Emitted from" = "linalg.generic"}
      memref.store %4, %arg0[%arg5] : memref<?xf64>
      scf.yield
    } {"Emitted from" = "linalg.generic"}
    return
  }
}
