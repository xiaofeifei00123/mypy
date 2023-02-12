program test

implicit none
integer :: fs
! integer :: pad_num
integer :: x,y  ! 数组维度,m为行数，n为列数
! integer :: i,j
real :: a(50, 50)
real :: b(50, 50)
real :: ft(11,11)

fs = 11
! pad_num = int(filter_size/2)

do x =1, 11
    do y = 1, 11
        ft(x,y) = 1
    end do
end do

do x =1, 50
    do y = 1, 50
        a(x,y) = 2
    end do
end do

a(3,30) = 5

x = 50
y = 50
call cal_ave(a, ft, x,y, fs,b)
print*, b
! print* ,a(10,:)
contains

subroutine cal_ave(array_in, filter_template, m, n, filter_size, array_out)
    ! argument type, intent(inout) :: cal_ave
    implicit none
    integer :: filter_size
    integer :: m, n  ! 数组维度,m为行数，n为列数
    integer :: pad_num
    real :: filter_template(filter_size, filter_size)
    real :: array_in(m, n)
    real,INTENT(OUT) :: array_out(m, n)
    integer :: i,j
    
    pad_num = int(filter_size/2) 
    do i = pad_num+1, m-pad_num, 1
        do j = pad_num+1, n-pad_num, 1
            ! print*, i-pad_num,i+pad_num
            ! b(i,j) =  1
            ! print*, shape(filter_template)
            ! print*, shape(a(i-pad_num:i+pad_num, j-pad_num,j+pad_num))
            array_out(i,j) = sum(filter_template*array_in(i-pad_num:i+pad_num:1, j-pad_num:j+pad_num:1))/(filter_size*filter_size)
            ! print*, array_out(i,j)
            ! print*, shape(filter_template)
            ! print*, shape(a(i-pad_num:i+pad_num:1, j-pad_num:j+pad_num:1))
            ! print*, i-pad_num
        end do
    end do

end subroutine cal_ave


end program test






! subroutine cal_ave3d(array_in, filter_template, array_out, x,y,z, filter_size)
!     implicit none
!     integer :: filter_size
!     integer :: x, y, z  ! 数组维度, x是多少列，y是多少行，z是多少层
!     real :: filter_template(filter_size, filter_size) ! 二维
!     real :: array_in(z, y, x)  ! 三维
!     !f2py real*4,intent(in) :: array_in
!     !f2py real*4,intent(in) :: filter_templates
!     !f2py real*4,intent(out) :: array_out
!     ! 对数组来说，in就是in, out就是out, 不要inout, 对于out的数，它就不需要传进来，作为返回值，返回
!     real :: array_out(z, y, x)
!     integer :: i, j,k  ! i是行，j是列, k是层
!     do k = 1, z, 1
!         call cal_ave2d(array_in(k,:,:), filter_template, array_out(k,:,:), y,x, filter_size)
!         ! call cal_ave2d_fortran(array_in(k,:,:), filter_template, array_out(k,:,:), y,x, filter_size)
!     end do
!     ! print*, sum(array_in(:,:,:)-array_out(:,:,:))
! end subroutine cal_ave3d



! subroutine cal_ave2d(array_in,filter_template,array_out, m, n, filter_size)
!     !
!     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     !example(python):
!     !def ftest(da, ftsize):
!     !    ft = np.ones((ftsize,ftsize), dtype=np.float32)
!     !    pad_num = int(ftsize/2)
!     !    daa = np.pad(da, ((pad_num, pad_num), (pad_num, pad_num)), mode="edge")  # 填充边界上的数
!     !    m,n = daa.shape # 获取填充后的输入图像的大小
!     !    back_da = calcc.caculate.cal_ave2d(daa, ft, m,n,ftsize)  # 传递参数时，数组得在前面
!     !    # print(np.sum(daa-back_da))
!     !    back_da_return = back_da[pad_num:m - pad_num, pad_num:n - pad_num]  # 裁剪
!     !    return back_da_return
!     !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     ! 二维均值滤波器，或者说是用区域平均值，代替原始值
!     implicit none
!     integer :: filter_size
!     integer :: m, n  ! 数组维度,m为行数，n为列数
!     integer :: pad_num
!     real*4 :: filter_template(filter_size, filter_size)   !滤波器
!     real*4 :: array_in(m, n)
!     real*4 :: array_out(m, n)
!     integer :: i,j
!     ! print*, '行数是',m, '列数是',n

!     !f2py real*4,intent(in) :: array_in
!     !f2py real*4,intent(in) :: filter_templates
!     !f2py real*4,intent(out) :: array_out
    
!     pad_num = int(filter_size/2) 
!     do i = pad_num+1, m-pad_num, 1
!         do j = pad_num+1, n-pad_num, 1
!             array_out(i,j) = sum(filter_template*array_in(i-pad_num:i+pad_num:1, j-pad_num:j+pad_num:1))/(filter_size*filter_size)
!             ! print*, array_out(i,j)
!         end do
!     end do
!     ! print*, sum(array_in-array_out)

! end subroutine cal_ave2d

