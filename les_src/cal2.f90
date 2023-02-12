module aaa


contains


    subroutine test(a)
        implicit none
        integer :: a
        !f2py integer,intent(in) :: a
        print*, a
    end subroutine test


end module aaa