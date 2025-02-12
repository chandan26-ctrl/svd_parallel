program svd_parallel
    use mpi
    use scalapack
    implicit none
    include 'mpif.h'
    integer :: ierr, myrank, numprocs, context, myrow, mycol
    integer :: m, n, mb, nb, descA(9), descU(9), descVT(9), info
    integer :: np_rows, np_cols, myrow_mpi, mycol_mpi
    double precision, allocatable :: A(:,:), U(:,:), VT(:,:), S(:)
    integer :: iam, nprocs
    
    call MPI_INIT(ierr)
    call MPI_COMM_RANK(MPI_COMM_WORLD, myrank, ierr)
    call MPI_COMM_SIZE(MPI_COMM_WORLD, numprocs, ierr)
    
    ! Matrix size and block size
    m = 3
    n = 3
    mb = 2  ! Block size for row distribution
    nb = 2  ! Block size for column distribution
    
    ! Process grid dimensions
    np_rows = 2
    np_cols = 2
    
    call blacs_get(-1, 0, context)
    call blacs_gridinit(context, 'Row-major', np_rows, np_cols)
    call blacs_gridinfo(context, np_rows, np_cols, myrow_mpi, mycol_mpi)
    
    allocate(A(m, n), U(m, m), VT(n, n), S(n))
    
    ! Initialize matrix A on rank 0
    if (myrank == 0) then
        A = reshape([1.0, 2.0, 3.0, &
                     4.0, 5.0, 6.0, &
                     7.0, 8.0, 9.0], [m, n])
    endif
    
    ! Distribute matrix across processes
    call pdgesvd('V', 'V', m, n, A, 1, 1, descA, S, U, 1, 1, descU, VT, 1, 1, descVT, info)
    
    if (info .ne. 0) then
        print *, 'SVD failed with info =', info
        stop
    endif
    
    ! Root process prints results
    if (myrank == 0) then
        print *, 'Singular values:'
        print *, S
        print *, 'Left singular vectors (U):'
        print *, U
        print *, 'Right singular vectors (V^T):'
        print *, VT
    endif
    
    ! Cleanup
    call blacs_gridexit(context)
    call MPI_FINALIZE(ierr)
    
end program svd_parallel
