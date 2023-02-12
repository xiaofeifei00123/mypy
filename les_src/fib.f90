subroutine FIB(A,N)
    INTEGER N
    REAL*4 A(N)
    do i = 1, N
        if (i.EQ.1) then
            A(i) = 0.0
        elseif (i.EQ.2) then
            A(i) = 1.0
       else
            A(i) = A(i-1)+A(i-2)
        endif
    enddo
end subroutine  FIB