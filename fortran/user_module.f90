!******************************************************************************
! MODULE: user_module
!******************************************************************************
!
! DESCRIPTION: 
!> @brief Module that contain user defined function. This module can call other module and subroutine. 
!! The only public routine is mfo_user, that return an acceleration that 
!! mimic a random force that depend on what the user want to model.
!
!******************************************************************************

module user_module

  use types_numeriques
  use mercury_globals

  implicit none

  private

  public :: mfo_user

  contains

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
!> @author 
!> John E. Chambers
!
!> @date 2 March 2001
!
! DESCRIPTION: 
!> @brief Applies an arbitrary force, defined by the user.
!!\n\n
!! If using with the symplectic algorithm MAL_MVS, the force should be
!! small compared with the force from the central object.
!! If using with the conservative Bulirsch-Stoer algorithm MAL_BS2, the
!! force should not be a function of the velocities.
!! \n\n
!! Code Units are in AU, days and solar mass * K2 (mass are in solar mass, but multiplied by K2 earlier in the code).
!
!> @note All coordinates and velocities must be with respect to central body
!
!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
  subroutine mfo_user (time,nbod,nbig,time_step,Rsth,Rsth5,Rsth10,rg2s,k2s,k2fs &
          ,sigmast,Rp,Rp5,Rp10,k2p,k2fp,rg2p,sigmap,m,x,v,spin,a,Torque)
!  m             = mass (in solar masses * K2)
!  x             = coordinates (x,y,z) with respect to the central body [AU]
!  v             = velocities (vx,vy,vz) with respect to the central body [AU/day]
!  nbod          = current number of bodies (INCLUDING the central object)
!  nbig          =    "       "    " big bodies (ones that perturb everything else)
!  time          = current epoch [days]

    use physical_constant
    use mercury_constant  
    use tides_constant_GR
    use orbital_elements, only : mco_x2el
    use spline

    implicit none

    ! Input
    integer, intent(in) :: nbod !< [in] current number of bodies (1: star; 2-nbig: big bodies; nbig+1-nbod: small bodies)
    integer, intent(in) :: nbig !< [in] current number of big bodies (ones that perturb everything else)
    real(double_precision), intent(in) :: time !< [in] current epoch (days)
    real(double_precision), intent(in) :: time_step !< [in] current epoch (days)
    real(double_precision), intent(in) :: m(nbod) !< [in] mass (in solar masses * K2)
    real(double_precision), intent(in) :: x(3,nbod)
    real(double_precision), intent(in) :: v(3,nbod)
    real(double_precision), intent(in) :: spin(3,nbod)
    real(double_precision), intent(in) :: Rp(10),Rp5(10),Rp10(10),k2p(10),k2fp(10),rg2p(10),sigmap(10)
    real(double_precision), intent(in) :: Rsth,Rsth5,Rsth10,rg2s,k2s,k2fs,sigmast
    
    ! Output
    real(double_precision),intent(out) :: a(3,nbod)
    real(double_precision),intent(out) :: Torque(3,nbod)
    
    !---------------------------------------------------------------------------
    !------Local-------

    ! Local
    integer :: j,kk, error, iPs0, nptmss
    integer :: flagrg2=0
    integer :: flagtime=0
    integer :: ispin=0
    integer :: iwrite=0
    integer :: charge_data = 0
    real(double_precision) :: flagbug=0.d0
    real(double_precision) :: timestep

    ! In case of BS integrator, we need time on previous timestep to know the
    ! current timestep
    real(double_precision) :: time_bf
    real(double_precision) :: output_tmp

    ! Temporary orbital elements needed to calculate pseudo-synchronization for planets:
    real(double_precision) :: gm,qq,ee,ii,pp,nn,ll
    real(double_precision), dimension(ntid+1) :: qa,ea,ia,pa,na,la

    ! Initial rotation and rotation of the star:
    real(double_precision) :: Pst0,Pst

    ! Total forces:
    real(double_precision) :: F_tid_tot_x,F_tid_tot_y,F_tid_tot_z
    real(double_precision) :: F_rot_tot_x,F_rot_tot_y,F_rot_tot_z
    real(double_precision) :: F_GR_tot_x,F_GR_tot_y,F_GR_tot_z
    !real(double_precision) :: acc_rot_x,acc_rot_y,acc_rot_z
    !real(double_precision) :: acc_GR_x,acc_GR_y,acc_GR_z

    ! Sum of the forces 
    real(double_precision) :: sum_F_tid_x,sum_F_tid_y,sum_F_tid_z
    real(double_precision) :: sum_F_rot_x,sum_F_rot_y,sum_F_rot_z
    real(double_precision) :: sum_F_GR_x,sum_F_GR_y,sum_F_GR_z

    ! Torques for planets and star:
    real(double_precision) :: N_tid_px,N_tid_py,N_tid_pz,N_tid_sx,N_tid_sy,N_tid_sz
    real(double_precision) :: N_rot_px,N_rot_py,N_rot_pz,N_rot_sx,N_rot_sy,N_rot_sz
    real(double_precision), dimension(3,nbig+1) :: Ns

    ! Runge Kutta terms (6 order):
    real(double_precision) :: k_rk_1x,k_rk_2x,k_rk_3x,k_rk_4x,k_rk_5x,k_rk_6x
    real(double_precision) :: k_rk_1y,k_rk_2y,k_rk_3y,k_rk_4y,k_rk_5y,k_rk_6y
    real(double_precision) :: k_rk_1z,k_rk_2z,k_rk_3z,k_rk_4z,k_rk_5z,k_rk_6z
    ! Runge Kutta terms (6 order) for position and velocity of the first half of mercury timestep:
    real(double_precision), dimension(3,nbig+1) :: xh_1_rk2,vh_1_rk2
    real(double_precision), dimension(3,nbig+1) :: xh_1_rk3,vh_1_rk3
    real(double_precision), dimension(3,nbig+1) :: xh_1_rk4,vh_1_rk4
    real(double_precision), dimension(3,nbig+1) :: xh_1_rk5,vh_1_rk5
    real(double_precision), dimension(3,nbig+1) :: xh_1_rk6,vh_1_rk6
    ! Runge Kutta terms (6 order) for position and velocity of the second half of mercury timestep:
    real(double_precision), dimension(3,nbig+1) :: xh_2_rk2,vh_2_rk2
    real(double_precision), dimension(3,nbig+1) :: xh_2_rk3,vh_2_rk3
    real(double_precision), dimension(3,nbig+1) :: xh_2_rk4,vh_2_rk4
    real(double_precision), dimension(3,nbig+1) :: xh_2_rk5,vh_2_rk5
    real(double_precision), dimension(3,nbig+1) :: xh_2_rk6,vh_2_rk6
    ! Values of spin, radius and moments of inertia (rg) used for Runge Kutta
    real(double_precision) :: spin0,spinp0,spinb0
    real(double_precision) :: Rst,Rst_5,Rst_10,Rstb
    !real(double_precision) :: Rsth,Rsth5,Rsth10,Rstbh
    real(double_precision) :: Rst0,Rst0_5,Rst0_10,Rstb0
    real(double_precision) :: rg2s0
    ! Position/velocity "before" (from previous steps) used for Runge Kutta::
    real(double_precision), dimension(3,10) :: xh_bf,vh_bf,xh_bf2,vh_bf2
    ! Temporary value for runge kutta:
    real(double_precision) :: xintermediate


    ! Temporary (tmp_dEdt is the derivative of the energy loss due to tides)
    real(double_precision) :: tmp,tmp1,tmp_dEdt

    ! Time related and temporary things:
    real(double_precision) :: dt,hdt
    real(double_precision), dimension(3) :: bobo

    ! Integration of the spin (total torque tides):
    real(double_precision), dimension(3) :: totftides
    real(double_precision), dimension(3) :: sum_RK

    ! Temporary accelerations for each effect:
    real(double_precision), dimension(3,nbig+1) :: a1,a2,a3
    ! Heliocentric coordinates:
    real(double_precision), dimension(3,nbig+1) :: xh,vh
    ! Orbital angular momentum vector and norm for each planet:
    real(double_precision), dimension(3,nbig+1) :: horb
    real(double_precision), dimension(nbig+1) :: horbn

    ! Spins:
    !real(double_precision), dimension(3,10) :: spin,spin_bf


    ! Parameters for the planets: physical radius
    !real(double_precision), dimension(10) :: Rp,Rp5,Rp10
    !! Parameters for the planets: dissipation of the planet, general relativity stuff in tintin, love number, time lag, moments of inertia
    !real(double_precision), dimension(10) :: sigmap,tintin,k2p,k2fp,k2pdeltap,rg2p
    real(double_precision), dimension(10) :: tintin

    ! Data tables for evolving host body:
    ! - Data for Brown dwarf
    real(double_precision), dimension(:), allocatable :: timeBD,radiusBD,lumiBD,HZinGJ,HZoutGJ,HZinb,HZoutb
    real(double_precision), dimension(37) :: rg2st,trg2,rg1,rg2,rg3,rg4,rg5,rg6,rg7,rg8,rg9,rg10,rg11,rg12
    !     * Initial rotation period for brown dwarfs (BD)
    real(double_precision), parameter, dimension(12) :: Ps0 = (/8.d0,13.d0,19.d0,24.d0,30.d0,36.d0,41.d0, &
         47.d0,53.d0,58.d0,64.d0,70.d0/)
    !     * Love number for BD
    real(double_precision), parameter, dimension(12) :: k2st = (/0.379d0,0.378d0,0.376d0,0.369d0, &
         0.355d0,0.342d0,0.333d0,0.325d0,0.311d0,0.308d0,0.307d0,0.307d0/)
    ! - Data for Star
    real(double_precision), dimension(2003) :: timestar,radiusstar,d2radiusstar
    ! - Data for M dwarf
    real(double_precision), dimension(1065) :: timedM,radiusdM
    ! - Data for Jupiter
    real(double_precision), dimension(4755) :: timeJup,radiusJup,k2Jup,rg2Jup,spinJup


    character(len=80) :: planet_spin_filename
    character(len=80) :: planet_orbt_filename
    character(len=80) :: planet_dEdt_filename

    ! Save data of tables for evolving host body
    ! - Data for Brown Dwarf
    save timeBD,radiusBD
    save trg2, rg2st
    ! - Date for Star
    save timestar,radiusstar,d2radiusstar
    ! - Data for M dwarf
    save timedM,radiusdM
    ! - Data for Jupiter
    save timeJup,radiusJup,k2Jup,rg2Jup,spinJup


    ! Save data for integration
    save xh_bf,vh_bf,xh_bf2,vh_bf2
    save timestep,nptmss
    save dt,hdt
    save tintin
    ! For BS integration
    save time_bf


    ! Flags
    save flagrg2,flagtime,ispin,flagbug,charge_data

    !------------------------------------------------------------------------------
    ! superzoyotte

    ! Acceleration initialization
    do j = 1, nbod
        a(1,j) = 0.d0
        a(2,j) = 0.d0
        a(3,j) = 0.d0
        Torque(1,j) = 0.d0
        Torque(2,j) = 0.d0
        Torque(3,j) = 0.d0
    end do
    do j=2,ntid+1
        a1(1,j) = 0.d0
        a1(2,j) = 0.d0
        a1(3,j) = 0.d0
        a2(1,j) = 0.d0
        a2(2,j) = 0.d0
        a2(3,j) = 0.d0
        a3(1,j) = 0.d0
        a3(2,j) = 0.d0
        a3(3,j) = 0.d0
        if (ispin.eq.0) then
            qq = 0.d0
            ee = 0.d0
            pp = 0.d0
            ii = 0.d0
            nn = 0.d0
            ll = 0.d0
            qa(j) = 0.d0
            ea(j) = 0.d0
            pa(j) = 0.d0
            ia(j) = 0.d0
            na(j) = 0.d0
            la(j) = 0.d0
        endif
    end do

    output_tmp = output
    !write(*,*) "input mfo_user",time,nbod,nbig,time_step,Rsth,Rsth5,Rsth10
    !write(*,*) "input mfo_user",rg2s,k2s,k2fs,sigmast
    !write(*,*) "input mfo_user",Rp(2),Rp5(2),Rp10(2),k2p(1),k2fp(1),rg2p(1),sigmap(2)
    !write(*,*) "input mfo_user",m
    !write(*,*) "input mfo_user",sqrt(x(1,2)*x(1,2)+x(2,2)*x(2,2)+x(3,2)*x(3,2))
    !write(*,*) "input mfo_user",x
    !write(*,*) "input mfo_user",v
    !write(*,*) "input mfo_user",spin
    !if (iwrite.eq.0) then 
        !call write_simus_properties()
        !iwrite = 1
    !endif

    ! Timestep calculation
    if (flagtime.eq.0) then 
        dt = time_step
        hdt = 0.5d0*dt
        flagtime = flagtime+1
        if (crash.eq.0) timestep = 0.0d0
        if (crash.eq.1) timestep = time + output * 365.25d0
    endif
    if (flagtime.ne.0) then
        dt = time - time_bf
        hdt = 0.5d0*dt
    endif

    ! Following calculations in heliocentric coordinates   
    !call conversion_dh2h(nbod,nbig,m,x,v,xh,vh)    
    do j=2,ntid+1
        xh(1,j) = x(1,j)
        xh(2,j) = x(2,j)
        xh(3,j) = x(3,j)
        vh(1,j) = v(1,j)
        vh(2,j) = v(2,j)
        vh(3,j) = v(3,j)
    enddo

    if (ispin.eq.0) then
        do j=2,ntid+1
            xh_bf(1,j) = xh(1,j)
            xh_bf(2,j) = xh(2,j)
            xh_bf(3,j) = xh(3,j)
            vh_bf(1,j) = vh(1,j)
            vh_bf(2,j) = vh(2,j)
            vh_bf(3,j) = vh(3,j)
        enddo
    endif      

    ! Definition of factor used for GR force calculation
    if ((GenRel.eq.1).and.(ispin.eq.0)) then
        tintin(1) = 0.
        do j=2,ntid+1
            tintin(j) = m(1)*m(j)/(m(1)+m(j))**2
        enddo
        do j=ntid+2,10
            tintin(j) = 0.
        enddo
    endif

    if (flagbug.ge.0) then

        ! If you have tides of rot flat
        if ((tides.eq.1).or.(rot_flat.eq.1)) then 
 
            ! Calculation of orbital angular momentum
            ! horb (without mass) in AU^2/day
            do j=2,ntid+1
                horb(1,j)  = (xh(2,j)*vh(3,j)-xh(3,j)*vh(2,j))
                horb(2,j)  = (xh(3,j)*vh(1,j)-xh(1,j)*vh(3,j))
                horb(3,j)  = (xh(1,j)*vh(2,j)-xh(2,j)*vh(1,j))    
                horbn(j) = sqrt(horb(1,j)*horb(1,j)+horb(2,j)*horb(2,j)+horb(3,j)*horb(3,j))
            end do

        endif   


        !---------------------------------------------------------------------------
        !---------------------------------------------------------------------------
        !------------------------------  Torque calculation  -----------------------
        !---------------------------------------------------------------------------
        !---------------------------------------------------------------------------

        if (flagbug.ge.0) then

            if ((tides.eq.1).or.(rot_flat.eq.1)) then       

                !---------------------------------------------------------------------
                !----------------------  STAR spin evolution  ------------------------
                !---------------------------------------------------------------------

                !---------------------------------------------------------------------
                !---------------------  First half of timestep  ---------------------- 
                !---------------------------------------------------------------------

                ! Torque at last time step t=time-dt
                ! Calculation of Runge-kutta factor k1
                do j=2,ntid+1 
                    if (tides.eq.1) then
                        call Torque_tides_s (nbod,m,xh_bf(1,j),xh_bf(2,j),xh_bf(3,j) &
                            ,vh_bf(1,j),vh_bf(2,j),vh_bf(3,j) &
                            ,spin(1,1),spin(2,1),spin(3,1) &
                            ,Rsth10,sigmast,j,N_tid_sx,N_tid_sy,N_tid_sz)

                    else
                        N_tid_sx = 0.0d0
                        N_tid_sy = 0.0d0
                        N_tid_sz = 0.0d0
                    endif
                    if (rot_flat.eq.1) then 
                        call Torque_rot_s (nbod,m,xh_bf(1,j),xh_bf(2,j),xh_bf(3,j) &
                             ,spin(1,1),spin(2,1),spin(3,1) &
                             ,Rsth5,k2fs,j,N_rot_sx,N_rot_sy,N_rot_sz)
                    else
                        N_rot_sx = 0.0d0
                        N_rot_sy = 0.0d0
                        N_rot_sz = 0.0d0
                    endif
                    Ns(1,j) =  tides*N_tid_sx + rot_flat*N_rot_sx
                    Ns(2,j) =  tides*N_tid_sy + rot_flat*N_rot_sy  
                    Ns(3,j) =  tides*N_tid_sz + rot_flat*N_rot_sz
                enddo
                totftides(1) = 0.d0
                totftides(2) = 0.d0
                totftides(3) = 0.d0
                do j=2,ntid+1 
                    tmp = m(1)/(m(1)+m(j))
                    totftides(1) = totftides(1) + tmp*Ns(1,j)
                    totftides(2) = totftides(2) + tmp*Ns(2,j)
                    totftides(3) = totftides(3) + tmp*Ns(3,j)
                end do   


                !---------------------------------------------------------------------
                !---------------------  PLANETS spin evolution  ----------------------
                !---------------------------------------------------------------------

         
                do j=2,ntid+1 
                    !tmp = - hdt*K2*m(1)/(m(j)*(m(j)+m(1))*rg2p(j-1)*Rp(j)*Rp(j))
                    tmp =  m(1)/(m(j)+m(1))

                    ! Torque at last time step t=time-dt
                    ! Calculation of Runge-kutta factor k1
                    if (tides.eq.1) then
                        call Torque_tides_p (nbod,m,xh_bf(1,j),xh_bf(2,j),xh_bf(3,j) &
                            ,vh_bf(1,j),vh_bf(2,j),vh_bf(3,j) &
                            ,spin(1,j),spin(2,j),spin(3,j),Rp10(j),sigmap(j),j &
                            ,N_tid_px,N_tid_py,N_tid_pz)

                        N_tid_px = tmp*N_tid_px
                        N_tid_py = tmp*N_tid_py
                        N_tid_pz = tmp*N_tid_pz 
                    else
                        N_tid_px = 0.0d0
                        N_tid_py = 0.0d0
                        N_tid_pz = 0.0d0 
                    endif
                    if (rot_flat.eq.1) then 
                        call Torque_rot_p (nbod,m,xh_bf(1,j),xh_bf(2,j),xh_bf(3,j) &
                             ,spin(1,j),spin(2,j),spin(3,j) &
                             ,Rp5(j),k2fp(j-1),j,N_rot_px,N_rot_py,N_rot_pz)
                        N_rot_px = tmp*N_rot_px
                        N_rot_py = tmp*N_rot_py
                        N_rot_pz = tmp*N_rot_pz 
                    else
                        N_rot_px = 0.0d0 
                        N_rot_py = 0.0d0 
                        N_rot_pz = 0.0d0  
                    endif

                enddo

                ! Define here torque 1: sum of torques on star, 2: torque on planet
                Torque(1:3,1) = (/totftides(1),totftides(2),totftides(3)/)
                Torque(1:3,2) = (/tides*N_tid_px + rot_flat*N_rot_px &
                    ,tides*N_tid_py + rot_flat*N_rot_py &
                    ,tides*N_tid_pz + rot_flat*N_rot_pz/)
            endif
        endif 

        !---------------------------------------------------------------------------
        !---------------------------------------------------------------------------
        !------------------------  ACCELERATIONS -----------------------------------
        !---------------------------------------------------------------------------
        !---------------------------------------------------------------------------


        ! Initialization of the sum of all the force along x, y and z
        ! For tides
        sum_F_tid_x = 0.0d0
        sum_F_tid_y = 0.0d0
        sum_F_tid_z = 0.0d0
        ! For rotation
        sum_F_rot_x = 0.0d0
        sum_F_rot_y = 0.0d0
        sum_F_rot_z = 0.0d0
        ! For GR
        sum_F_GR_x = 0.0d0
        sum_F_GR_y = 0.0d0
        sum_F_GR_z = 0.0d0

        ! The acceleration in the heliocentric coordinate is not just F/m,
        ! it must be the reduced mass, and the effect of other planets on the
        ! star also has to be removed, see in article for explanation

        do j=2,ntid+1 

            if (tides.eq.1) then 
                tmp  = K2/m(j)
                tmp1 = K2/m(1) 
                ! Here I call subroutine that gives the total tidal force
                call F_tides_tot (nbod,m,xh(1,j),xh(2,j),xh(3,j),vh(1,j),vh(2,j),vh(3,j),spin &
                     ,Rsth5,Rsth10,k2s,sigmast &
                     ,Rp5(j),Rp10(j),k2fp(j-1),k2p(j-1),sigmap(j) &
                     ,j,F_tid_tot_x,F_tid_tot_y,F_tid_tot_z)
                ! Calculation of the acceleration     
                a1(1,j) = tmp*F_tid_tot_x
                a1(2,j) = tmp*F_tid_tot_y
                a1(3,j) = tmp*F_tid_tot_z
                ! Calculation of the sum of the acceleration of all planets
                sum_F_tid_x = sum_F_tid_x + tmp1*F_tid_tot_x
                sum_F_tid_y = sum_F_tid_y + tmp1*F_tid_tot_y
                sum_F_tid_z = sum_F_tid_z + tmp1*F_tid_tot_z
            else
                a1(1,j) = 0.0d0
                a1(2,j) = 0.0d0
                a1(3,j) = 0.0d0
            endif

            if (rot_flat.eq.1) then 
                tmp  = K2/m(j)
                tmp1 = K2/m(1) 
                ! Here I call subroutine that gives the total rotation flattening
                ! induced force
                call F_rotation (nbod,m,xh(1,j),xh(2,j),xh(3,j),spin &
                     ,Rsth5,k2fs,Rp5(j),k2fp(j-1) &
                     ,j,F_rot_tot_x,F_rot_tot_y,F_rot_tot_z)
                ! Calculation of the acceleration     
                a2(1,j) = tmp*F_rot_tot_x
                a2(2,j) = tmp*F_rot_tot_y
                a2(3,j) = tmp*F_rot_tot_z
                ! Calculation of the sum of the acceleration of all planets
                sum_F_rot_x = sum_F_rot_x + tmp1*F_rot_tot_x
                sum_F_rot_y = sum_F_rot_y + tmp1*F_rot_tot_y
                sum_F_rot_z = sum_F_rot_z + tmp1*F_rot_tot_z
            else
                a2(1,j) = 0.0d0
                a2(2,j) = 0.0d0
                a2(3,j) = 0.0d0
            endif

            if (GenRel.eq.1) then 
                tmp  = K2/m(j)
                tmp1 = K2/m(1) 
                ! Here I call subroutine that gives the total GR force
                call F_GenRel (nbod,m,xh(1,j),xh(2,j),xh(3,j),vh(1,j),vh(2,j),vh(3,j) &
                     ,tintin(j),C2,j,F_GR_tot_x,F_GR_tot_y,F_GR_tot_z)
                ! Calculation of the acceleration     
                a3(1,j) = tmp*F_GR_tot_x
                a3(2,j) = tmp*F_GR_tot_y
                a3(3,j) = tmp*F_GR_tot_z
                ! Calculation of the sum of the acceleration of all planets
                sum_F_GR_x = sum_F_GR_x + tmp1*F_GR_tot_x
                sum_F_GR_y = sum_F_GR_y + tmp1*F_GR_tot_y
                sum_F_GR_z = sum_F_GR_z + tmp1*F_GR_tot_z
            else
                a3(1,j) = 0.0d0
                a3(2,j) = 0.0d0
                a3(3,j) = 0.0d0
            endif
        end do

        ! Sum of all accelerations
        do j=2,ntid+1
            a(1,j) = tides*(a1(1,j)+sum_F_tid_x)+rot_flat*(a2(1,j)+sum_F_rot_x)+GenRel*(a3(1,j)+sum_F_GR_x)
            a(2,j) = tides*(a1(2,j)+sum_F_tid_y)+rot_flat*(a2(2,j)+sum_F_rot_y)+GenRel*(a3(2,j)+sum_F_GR_y)
            a(3,j) = tides*(a1(3,j)+sum_F_tid_z)+rot_flat*(a2(3,j)+sum_F_rot_z)+GenRel*(a3(3,j)+sum_F_GR_z)
        end do

        !---------------------------------------------------------------------------
        !---------------------------------------------------------------------------
        !-------------------------  Write spin and stuff  --------------------------
        !---------------------------------------------------------------------------
        !---------------------------------------------------------------------------

        if ((flagbug.ge.1).and.((tides.eq.1).or.(rot_flat.eq.1))) then          
            if (crash.eq.0) then
                if (time.ge.timestep) then
                    open(13, file="spins.out", access="append")
                    write(13,'(8("  ", es20.10e3))') time/365.25d0,spin(1,1),spin(2,1),spin(3,1),Rsth/rsun,rg2s,k2s,sigmast
                    close(13)
                    do j=2,ntid+1
                        write(planet_spin_filename,('(a,i1,a)')) 'spinp',j-1,'.out'
                        write(planet_orbt_filename,('(a,i1,a)')) 'horb',j-1,'.out'
                        !write(planet_dEdt_filename,('(a,i1,a)')) 'dEdt',j-1,'.out'
                        open(13, file=planet_spin_filename, access='append')
                        write(13,'(6("  ", es20.10e3))') time/365.25d0,spin(1,j),spin(2,j),spin(3,j),Rp(j)/rsun,rg2p(j-1)
                        close(13)
                        open(13, file=planet_orbt_filename, access='append')
                        write(13,'(4("  ", es20.10e3))') time/365.25d0,horb(1,j),horb(2,j),horb(3,j)
                        close(13)
                        !if (tides.eq.1) then
                            !! Here I calculate the instantaneous energy loss in Msun.AU^2.day^-3
                            !call dEdt_tides (nbod,m,xh(1,j),xh(2,j),xh(3,j),vh(1,j),vh(2,j),vh(3,j),spin &
                                 !,Rp10(j),sigmap(j),j,tmp_dEdt)
                        !endif
                        !open(13, file=planet_dEdt_filename, access='append')
                        !write(13,'(4("  ", es20.10e3))') time/365.25d0,tmp_dEdt
                        !close(13)
                        !open(13, file='acceleration.out', access='append')
                        !write(13,'(4("  ", es20.10e3))') time/365.25d0, a(1,j),a(2,j),a(3,j)
                        !close(13)
                    enddo
                    if (time/365.25d0.ge.10.d0*output_tmp) then
                        output_tmp = output_tmp * 10.d0
                        timestep = timestep + output_tmp*365.25d0
                    else 
                        timestep = timestep + output_tmp*365.25d0
                    endif
                endif
            endif
            if (crash.eq.1) then
                if (time.ge.timestep) then
                    open(13, file="spins.out", access="append")
                    write(13,'(8("  ", es20.10e3))') time/365.25d0,spin(1,1),spin(2,1),spin(3,1),Rsth/rsun,rg2s,k2s,sigmast
                    close(13)
                    do j=2,ntid+1
                        write(planet_spin_filename,('(a,i1,a)')) 'spinp',j-1,'.out'
                        write(planet_orbt_filename,('(a,i1,a)')) 'horb',j-1,'.out'
                        !write(planet_dEdt_filename,('(a,i1,a)')) 'dEdt',j-1,'.out'
                        open(13, file=planet_spin_filename, access='append')
                        write(13,'(6("  ", es20.10e3))') time/365.25d0,spin(1,j),spin(2,j),spin(3,j),Rp(j)/rsun,rg2p(j-1)
                        close(13)
                        open(13, file=planet_orbt_filename, access='append')
                        write(13,'(4("  ", es20.10e3))') time/365.25d0,horb(1,j),horb(2,j),horb(3,j)
                        close(13)
                        !if (tides.eq.1) then
                            !! Here I calculate the instantaneous energy loss in Msun.AU^2.day^-3
                            !call dEdt_tides (nbod,m,xh(1,j),xh(2,j),xh(3,j),vh(1,j),vh(2,j),vh(3,j),spin &
                                 !,Rp10(j),sigmap(j),j,tmp_dEdt)
                        !endif
                        !open(13, file=planet_dEdt_filename, access='append')
                        !write(13,'(4("  ", es20.10e3))') time/365.25d0,tmp_dEdt
                        !close(13)
                        !open(13, file='acceleration.out', access='append')
                        !write(13,'(4("  ", es20.10e3))') time/365.25d0, a(1,j),a(2,j),a(3,j)
                        !close(13)
                    enddo
                    if (time/365.25d0.ge.10.d0*output_tmp) then
                        output_tmp = output_tmp * 10.d0
                        timestep = timestep + output_tmp*365.25d0
                    else 
                        timestep = timestep + output_tmp*365.25d0
                    endif

                endif
            endif
        endif    

    endif
    
    ! Attribution of new values (correponding at t) to old one (t-dt)
    do j=2,ntid+1 
        xh_bf2(1,j) = xh_bf(1,j)
        xh_bf2(2,j) = xh_bf(2,j)
        xh_bf2(3,j) = xh_bf(3,j)
        vh_bf2(1,j) = vh_bf(1,j)
        vh_bf2(2,j) = vh_bf(2,j)
        vh_bf2(3,j) = vh_bf(3,j)
        xh_bf(1,j) = xh(1,j)
        xh_bf(2,j) = xh(2,j)
        xh_bf(3,j) = xh(3,j)
        vh_bf(1,j) = vh(1,j)
        vh_bf(2,j) = vh(2,j)
        vh_bf(3,j) = vh(3,j)
    enddo


    if (algor.eq.2) time_bf = time

    Rst0    = Rsth
    Rst0_5  = Rsth5
    Rst0_10 = Rsth10
    rg2s0   = rg2s

    ! ispin is called once only during the whole time of the simulation
    if (flagbug.ge.2) ispin=1
    ! flagbug increases of each timestep, I don't know if I will keep that
    flagbug = flagbug+1

    return
  end subroutine mfo_user


  !-----------------------------------------------------------------------------
  !-----------------------------------------------------------------------------
  !-----------------------------------------------------------------------------
  !-----------------------------------------------------------------------------
  ! -----------------------------  SUBROUTINES ---------------------------------
  !-----------------------------------------------------------------------------
  !-----------------------------------------------------------------------------
  !-----------------------------------------------------------------------------
  !-----------------------------------------------------------------------------


  !-----------------------------------------------------------------------------
  !-----------------------------------------------------------------------------
  !----------------------------   GENERAL ONES  -------------------------------- 
  !-----------------------------------------------------------------------------
  !-----------------------------------------------------------------------------


  !-----------------------------------------------------------------------------
  ! Conversion from democratic heliocentric to heliocentric coordinates
  subroutine conversion_dh2h (nbod,nbig,m,x,v,xh,vh)
 
      implicit none
      ! Input/Output
      integer,intent(in) :: nbod,nbig
      real(double_precision),intent(in) :: m(nbod),x(3,nbod),v(3,nbod)
      real(double_precision), intent(out) :: xh(3,nbod),vh(3,nbod)
      ! Local
      integer :: j
      real(double_precision) :: mvsum(3),temp
      !-------------------------------------------------------------------------
      mvsum(1) = 0.d0
      mvsum(2) = 0.d0
      mvsum(3) = 0.d0
      do j = 2, nbod
          xh(1,j) = x(1,j)
          xh(2,j) = x(2,j)
          xh(3,j) = x(3,j)
          mvsum(1) = mvsum(1)  +  m(j) * v(1,j)
          mvsum(2) = mvsum(2)  +  m(j) * v(2,j)
          mvsum(3) = mvsum(3)  +  m(j) * v(3,j)
      end do
      temp = 1.d0 / m(1)
      mvsum(1) = temp * mvsum(1)
      mvsum(2) = temp * mvsum(2)
      mvsum(3) = temp * mvsum(3)
      do j = 2, nbod
          vh(1,j) = v(1,j) + mvsum(1)
          vh(2,j) = v(2,j) + mvsum(2)
          vh(3,j) = v(3,j) + mvsum(3)
      end do
      !-------------------------------------------------------------------------
      return
  end subroutine conversion_dh2h

subroutine conversion_h2dh (time,nbod,nbig,m,xh,vh,x,v)


  implicit none


  ! Input/Output
  integer, intent(in) :: nbod !< [in] current number of bodies (1: star; 2-nbig: big bodies; nbig+1-nbod: small bodies)
  integer, intent(in) :: nbig !< [in] current number of big bodies (ones that perturb everything else)
  real(double_precision),intent(in) :: time !< [in] current epoch (days)
  real(double_precision),intent(in) :: m(nbod) !< [in] mass (in solar masses * K2)
  real(double_precision),intent(in) :: xh(3,nbod) !< [in] coordinates (x,y,z) with respect to the central body (AU)
  real(double_precision),intent(in) :: vh(3,nbod) !< [in] velocities (vx,vy,vz) with respect to the central body (AU/day)

  real(double_precision),intent(out) :: v(3,nbod)
  real(double_precision),intent(out) :: x(3,nbod)

  ! Local
  integer :: j
  real(double_precision) :: mtot,temp,mvsum(3)

  !------------------------------------------------------------------------------

  mtot = 0.d0
  mvsum(1) = 0.d0
  mvsum(2) = 0.d0
  mvsum(3) = 0.d0

  do j = 2, nbod
     x(1,j) = xh(1,j)
     x(2,j) = xh(2,j)
     x(3,j) = xh(3,j)
     mtot = mtot + m(j)
     mvsum(1) = mvsum(1)  +  m(j) * vh(1,j)
     mvsum(2) = mvsum(2)  +  m(j) * vh(2,j)
     mvsum(3) = mvsum(3)  +  m(j) * vh(3,j)
  end do

  temp = 1.d0 / (m(1) + mtot)

  mvsum(1) = temp * mvsum(1)
  mvsum(2) = temp * mvsum(2)
  mvsum(3) = temp * mvsum(3)

  do j = 2, nbod
     v(1,j) = vh(1,j) - mvsum(1)
     v(2,j) = vh(2,j) - mvsum(2)
     v(3,j) = vh(3,j) - mvsum(3)
  end do

  !------------------------------------------------------------------------------

  return
end subroutine conversion_h2dh



  !-----------------------------------------------------------------------------
  ! Calculation of r(j), powers of r(j)
  ! Distances in AU
  subroutine rad_power (xhx,xhy,xhz,r_2,rr,r_4,r_5,r_7,r_8)
 
      implicit none
      ! Input/Output
      real(double_precision),intent(in) :: xhx,xhy,xhz
      real(double_precision), intent(out) :: r_2,rr,r_4,r_5,r_7,r_8
      !-------------------------------------------------------------------------
      r_2 = xhx*xhx+xhy*xhy+xhz*xhz
      rr  = sqrt(r_2)
      r_4 = r_2*r_2
      r_5 = r_4*rr
      r_7 = r_4*r_2*rr
      r_8 = r_4*r_4
      !-------------------------------------------------------------------------
      return
  end subroutine rad_power

  !-----------------------------------------------------------------------------
  ! Calculation of velocity vv(j), radial velocity vrad(j)
  ! velocities in AU/day
  subroutine velocities (xhx,xhy,xhz,vhx,vhy,vhz,v_2,norm_v,v_rad)

      implicit none
      ! Input/Output
      real(double_precision),intent(in) :: xhx,xhy,xhz,vhx,vhy,vhz
      real(double_precision), intent(out) :: v_2,norm_v,v_rad
      !-------------------------------------------------------------------------
      v_2    = vhx*vhx+vhy*vhy+vhz*vhz
      norm_v = sqrt(v_2)         
      ! Radial velocity
      v_rad  = (xhx*vhx+xhy*vhy+xhz*vhz)/sqrt(xhx*xhx+xhy*xhy+xhz*xhz)
      !-------------------------------------------------------------------------
      return
  end subroutine velocities

  !-----------------------------------------------------------------------------
  ! Calculation of r scalar spin
  subroutine r_scal_spin (xhx,xhy,xhz,spinx,spiny,spinz,rscalspin)
 
      implicit none
      ! Input/Output
      real(double_precision),intent(in) :: xhx,xhy,xhz
      real(double_precision),intent(in) :: spinx,spiny,spinz
      real(double_precision), intent(out) :: rscalspin
      !-------------------------------------------------------------------------
      rscalspin = xhx*spinx+xhy*spiny+xhz*spinz
      !-------------------------------------------------------------------------
      return
  end subroutine r_scal_spin

  !-----------------------------------------------------------------------------
  ! Calculation of the norm square of the spin
  subroutine norm_spin_2 (spinx,spiny,spinz,normspin_2)

      implicit none
      ! Input/Output
      real(double_precision),intent(in) :: spinx,spiny,spinz
      real(double_precision), intent(out) :: normspin_2
      !-------------------------------------------------------------------------
      normspin_2 = spinx*spinx+spiny*spiny+spinz*spinz
      !-------------------------------------------------------------------------
      return
  end subroutine norm_spin_2
  
  !-----------------------------------------------------------------------------
  !-----------------------------------------------------------------------------
  !-----------------------------  TIDES  ---------------------------------------
  !-----------------------------------------------------------------------------
  !-----------------------------------------------------------------------------

  !-----------------------------------------------------------------------------
  !--------------------------  tidal force  ------------------------------------
  ! Ftidr in Msun.AU.day-2
  ! Ftidos and Ftidop in Msun.AU.day-1
  ! K2 = G in AU^3.Msun-1.day-2

  !-----------------------------------------------------------------------------
  ! Conservative part of the radial tidal force
  subroutine F_tides_rad_cons (nbod,m,xhx,xhy,xhz,vhx,vhy,vhz &
         ,R_star5,k2_star &
         ,R_plan5,k2f_plan,j,Ftidr_cons)

      use physical_constant
      implicit none
      ! Input/Output
      integer,intent(in) :: nbod,j
      real(double_precision),intent(in) :: xhx,xhy,xhz,vhx,vhy,vhz
      real(double_precision),intent(in) :: R_star5,k2_star
      real(double_precision),intent(in) :: R_plan5,k2f_plan
      real(double_precision),intent(in) :: m(nbod)
      real(double_precision), intent(out) :: Ftidr_cons
      ! Local
      real(double_precision) :: tmp1,tmp2
      real(double_precision) :: r_2,rr,r_4,r_5,r_7,r_8
      !-------------------------------------------------------------------------
      call rad_power (xhx,xhy,xhz,r_2,rr,r_4,r_5,r_7,r_8)
      tmp1 = m(1)*m(1)
      tmp2 = m(j)*m(j)
      Ftidr_cons =  -3.0d0/(r_7*K2) &
        *(tmp2*R_star5*k2_star+tmp1*R_plan5*k2f_plan)

      !-------------------------------------------------------------------------
      return
  end subroutine F_tides_rad_cons

  !-----------------------------------------------------------------------------
  ! Dissipative part of the radial tidal force
  subroutine F_tides_rad_diss (nbod,m,xhx,xhy,xhz,vhx,vhy,vhz &
         ,R_star10,sigma_star &
         ,R_plan10,sigma_plan,j,Ftidr_diss)

      use physical_constant
      implicit none
      ! Input/Output
      integer,intent(in) :: nbod,j
      real(double_precision),intent(in) :: xhx,xhy,xhz,vhx,vhy,vhz
      real(double_precision),intent(in) :: R_star10,sigma_star
      real(double_precision),intent(in) :: R_plan10,sigma_plan
      real(double_precision),intent(in) :: m(nbod)
      real(double_precision), intent(out) :: Ftidr_diss
      ! Local
      real(double_precision) :: tmp,tmp1,tmp2
      real(double_precision) :: r_2,rr,r_4,r_5,r_7,r_8,v_2,norm_v,v_rad
      !-------------------------------------------------------------------------
      call rad_power (xhx,xhy,xhz,r_2,rr,r_4,r_5,r_7,r_8)
      call velocities (xhx,xhy,xhz,vhx,vhy,vhz,v_2,norm_v,v_rad)
      tmp  = K2*K2
      tmp1 = m(1)*m(1)
      tmp2 = m(j)*m(j)
      Ftidr_diss =  - 13.5d0*v_rad/(r_8*tmp) &
                *(tmp2*R_star10*sigma_star &
                +tmp1*R_plan10*sigma_plan)              
      !-------------------------------------------------------------------------
      return
  end subroutine F_tides_rad_diss

  !-----------------------------------------------------------------------------
  ! Sum of the dissipative and conservative part of the radial force
  subroutine F_tides_rad (nbod,m,xhx,xhy,xhz,vhx,vhy,vhz &
         ,R_star5,R_star10,k2_star,sigma_star &
         ,R_plan5,R_plan10,k2f_plan,k2_plan,sigma_plan,j,Ftidr)

      use physical_constant
      implicit none
      ! Input/Output
      integer,intent(in) :: nbod,j
      real(double_precision),intent(in) :: xhx,xhy,xhz,vhx,vhy,vhz
      real(double_precision),intent(in) :: sigma_star,sigma_plan
      real(double_precision),intent(in) :: R_star5,R_star10,k2_star
      real(double_precision),intent(in) :: R_plan5,R_plan10,k2f_plan,k2_plan
      real(double_precision),intent(in) :: m(nbod)
      real(double_precision), intent(out) :: Ftidr
      ! Local
      real(double_precision) :: Ftidr_cons,Ftidr_diss
      !-------------------------------------------------------------------------
      call F_tides_rad_cons (nbod,m,xhx,xhy,xhz,vhx,vhy,vhz &
         ,R_star5,k2_star &
         ,R_plan5,k2f_plan,j,Ftidr_cons)
      call F_tides_rad_diss (nbod,m,xhx,xhy,xhz,vhx,vhy,vhz &
         ,R_star10,sigma_star &
         ,R_plan10,sigma_plan,j,Ftidr_diss)
 
      Ftidr = Ftidr_cons + Ftidr_diss            
      !-------------------------------------------------------------------------
      return
  end subroutine F_tides_rad

  !-----------------------------------------------------------------------------
  ! Orthoradial part of the force due to stellar tides
  subroutine F_tides_ortho_star (nbod,m,xhx,xhy,xhz,R_star10 &
         ,sigma_star,j,Ftidos)
  
      use physical_constant
      implicit none
      ! Input/Output
      integer,intent(in) :: nbod,j
      real(double_precision),intent(in) :: xhx,xhy,xhz
      real(double_precision),intent(in) :: R_star10,sigma_star
      real(double_precision),intent(in) :: m(nbod)
      real(double_precision), intent(out) :: Ftidos
      ! Local
      real(double_precision) :: r_2,rr,r_4,r_5,r_7,r_8
      !-------------------------------------------------------------------------
      call rad_power (xhx,xhy,xhz,r_2,rr,r_4,r_5,r_7,r_8)

      Ftidos = 4.5d0*m(j)*m(j)*R_star10*sigma_star/(K2*K2*r_7)
      !-------------------------------------------------------------------------
      return
  end subroutine F_tides_ortho_star 

  !-----------------------------------------------------------------------------
  ! Orthoradial part of the force due to planetary tides
  subroutine F_tides_ortho_plan (nbod,m,xhx,xhy,xhz,R_plan10,sigma_plan &
         ,j,Ftidop)
  
      use physical_constant
      implicit none
      ! Input/Output
      integer,intent(in) :: nbod,j
      real(double_precision),intent(in) :: xhx,xhy,xhz
      real(double_precision),intent(in) :: R_plan10,sigma_plan
      real(double_precision),intent(in) :: m(nbod)
      real(double_precision), intent(out) :: Ftidop
      ! Local
      real(double_precision) :: r_2,rr,r_4,r_5,r_7,r_8
      !-------------------------------------------------------------------------
      call rad_power (xhx,xhy,xhz,r_2,rr,r_4,r_5,r_7,r_8)

      Ftidop = 4.5d0*m(1)*m(1)*R_plan10*sigma_plan/(K2*K2*r_7)
      !-------------------------------------------------------------------------
      return
  end subroutine F_tides_ortho_plan   

  !-----------------------------------------------------------------------------
  ! Torque on planet created by star 
  subroutine Torque_tides_p (nbod,m,xhx,xhy,xhz,vhx,vhy,vhz,spinx,spiny,spinz &
       ,R_plan10,sigma_plan,j,N_tid_px,N_tid_py,N_tid_pz)
  
      implicit none
      ! Input/Output
      integer,intent(in) :: nbod,j
      real(double_precision),intent(in) :: xhx,xhy,xhz,vhx,vhy,vhz
      real(double_precision),intent(in) :: spinx,spiny,spinz
      real(double_precision),intent(in) :: R_plan10,sigma_plan
      real(double_precision),intent(in) :: m(nbod)
      real(double_precision), intent(out) :: N_tid_px,N_tid_py,N_tid_pz
      ! Local
      real(double_precision) :: r_2,rr,r_4,r_5,r_7,r_8,rscalspin,Ftidop
      !-------------------------------------------------------------------------
      call rad_power (xhx,xhy,xhz,r_2,rr,r_4,r_5,r_7,r_8)
      call r_scal_spin (xhx,xhy,xhz,spinx,spiny,spinz,rscalspin)
      call F_tides_ortho_plan (nbod,m,xhx,xhy,xhz,R_plan10,sigma_plan,j,Ftidop)        

      N_tid_px = Ftidop*(rr*spinx-rscalspin*xhx/rr-1.0d0/rr*(xhy*vhz-xhz*vhy))
      N_tid_py = Ftidop*(rr*spiny-rscalspin*xhy/rr-1.0d0/rr*(xhz*vhx-xhx*vhz))
      N_tid_pz = Ftidop*(rr*spinz-rscalspin*xhz/rr-1.0d0/rr*(xhx*vhy-xhy*vhx))              
      !-------------------------------------------------------------------------
      return
  end subroutine Torque_tides_p 

  !-----------------------------------------------------------------------------
  ! Torque on star created by planet 
  subroutine Torque_tides_s (nbod,m,xhx,xhy,xhz,vhx,vhy,vhz,spinx,spiny,spinz &
       ,R_star10,sigma_star,j,N_tid_sx,N_tid_sy,N_tid_sz)
  
      implicit none
      ! Input/Output
      integer,intent(in) :: nbod,j
      real(double_precision),intent(in) :: xhx,xhy,xhz,vhx,vhy,vhz
      real(double_precision),intent(in) :: spinx,spiny,spinz
      real(double_precision),intent(in) :: R_star10,sigma_star
      real(double_precision),intent(in) :: m(nbod)
      real(double_precision), intent(out) :: N_tid_sx,N_tid_sy,N_tid_sz
      ! Local
      real(double_precision) :: r_2,rr,r_4,r_5,r_7,r_8,rscalspin,Ftidos
      !-------------------------------------------------------------------------
      call rad_power (xhx,xhy,xhz,r_2,rr,r_4,r_5,r_7,r_8)
      call r_scal_spin (xhx,xhy,xhz,spinx,spiny,spinz,rscalspin)
      call F_tides_ortho_star (nbod,m,xhx,xhy,xhz,R_star10 &
              ,sigma_star,j,Ftidos)          

      N_tid_sx = Ftidos*(rr*spinx-rscalspin*xhx/rr-1.0d0/rr*(xhy*vhz-xhz*vhy))
      N_tid_sy = Ftidos*(rr*spiny-rscalspin*xhy/rr-1.0d0/rr*(xhz*vhx-xhx*vhz))
      N_tid_sz = Ftidos*(rr*spinz-rscalspin*xhz/rr-1.0d0/rr*(xhx*vhy-xhy*vhx))              
      !-------------------------------------------------------------------------
      return
  end subroutine Torque_tides_s 

  !-----------------------------------------------------------------------------
  ! Total tidal force
  subroutine F_tides_tot (nbod,m,xhx,xhy,xhz,vhx,vhy,vhz,spin &
    ,R_star5,R_star10,k2_star,sigma_star &
    ,R_plan5,R_plan10,k2f_plan,k2_plan,sigma_plan,j &
    ,F_tid_tot_x,F_tid_tot_y,F_tid_tot_z)

    use physical_constant
    implicit none
    ! Input/Output
    integer,intent(in) :: nbod,j
    real(double_precision),intent(in) :: xhx,xhy,xhz,vhx,vhy,vhz
    real(double_precision),intent(in) :: sigma_star,sigma_plan
    real(double_precision),intent(in) :: R_star5,R_star10,k2_star
    real(double_precision),intent(in) :: R_plan5,R_plan10,k2f_plan,k2_plan
    real(double_precision),intent(in) :: m(nbod),spin(3,nbod)
    real(double_precision), intent(out) :: F_tid_tot_x,F_tid_tot_y,F_tid_tot_z
    ! Local
    real(double_precision) :: tmp1,r_2,rr,r_4,r_5,r_7,r_8,v_2,norm_v,v_rad
    real(double_precision) :: Ftidr,Ftidos,Ftidop
    !-------------------------------------------------------------------------
    call F_tides_rad (nbod,m,xhx,xhy,xhz,vhx,vhy,vhz &
       ,R_star5,R_star10,k2_star,sigma_star &
       ,R_plan5,R_plan10,k2f_plan,k2_plan,sigma_plan,j,Ftidr)
    call F_tides_ortho_star (nbod,m,xhx,xhy,xhz,R_star10 &
       ,sigma_star,j,Ftidos)
    call F_tides_ortho_plan (nbod,m,xhx,xhy,xhz,R_plan10,sigma_plan,j,Ftidop)
    call velocities (xhx,xhy,xhz,vhx,vhy,vhz,v_2,norm_v,v_rad)
    call rad_power (xhx,xhy,xhz,r_2,rr,r_4,r_5,r_7,r_8)

    tmp1 = Ftidr+(Ftidos+Ftidop)*v_rad/rr

    F_tid_tot_x = (tmp1*xhx/rr &
        + Ftidos/rr*(spin(2,1)*xhz-spin(3,1)*xhy-vhx) &
        + Ftidop/rr*(spin(2,j)*xhz-spin(3,j)*xhy-vhx))
    F_tid_tot_y = (tmp1*xhy/rr &
        + Ftidos/rr*(spin(3,1)*xhx-spin(1,1)*xhz-vhy) &
        + Ftidop/rr*(spin(3,j)*xhx-spin(1,j)*xhz-vhy))
    F_tid_tot_z = (tmp1*xhz/rr &
        + Ftidos/rr*(spin(1,1)*xhy-spin(2,1)*xhx-vhz) &
        + Ftidop/rr*(spin(1,j)*xhy-spin(2,j)*xhx-vhz)) 


      !-------------------------------------------------------------------------
      return
  end subroutine F_tides_tot

  !-----------------------------------------------------------------------------
  ! Instantaneous energy loss dE/dt due to tides
  ! in Msun.AU^2.day^(-3)
  subroutine dEdt_tides (nbod,m,xhx,xhy,xhz,vhx,vhy,vhz,spin &
         ,R_plan10,sigma_plan,j,dEdt)

      use physical_constant
      implicit none
      ! Input/Output
      integer,intent(in) :: nbod,j
      real(double_precision),intent(in) :: xhx,xhy,xhz,vhx,vhy,vhz
      real(double_precision),intent(in) :: R_plan10,sigma_plan
      real(double_precision),intent(in) :: m(nbod),spin(3,10)
      real(double_precision), intent(out) :: dEdt
      ! Local
      real(double_precision) :: tmp,tmp1
      real(double_precision) :: Ftidr_diss,Ftidop
      real(double_precision) :: r_2,rr,r_4,r_5,r_7,r_8,v_2,norm_v,v_rad
      real(double_precision) :: spinx,spiny,spinz,N_tid_px,N_tid_py,N_tid_pz
      !-------------------------------------------------------------------------
      call F_tides_rad_diss (nbod,m,xhx,xhy,xhz,vhx,vhy,vhz &
         ,0.0d0,0.0d0 &
         ,R_plan10,sigma_plan,j,Ftidr_diss)
      call F_tides_ortho_plan (nbod,m,xhx,xhy,xhz,R_plan10,sigma_plan,j,Ftidop)
      spinx = spin(1,j)
      spiny = spin(2,j)
      spinz = spin(3,j)
      call Torque_tides_p (nbod,m,xhx,xhy,xhz,vhx,vhy,vhz,spinx,spiny,spinz &
         ,R_plan10,sigma_plan,j,N_tid_px,N_tid_py,N_tid_pz)
      call velocities (xhx,xhy,xhz,vhx,vhy,vhz,v_2,norm_v,v_rad)
      call rad_power (xhx,xhy,xhz,r_2,rr,r_4,r_5,r_7,r_8)

      tmp = Ftidop/rr
      tmp1 = 1.d0/rr*(Ftidr_diss + tmp*v_rad)

      dEdt = -(tmp1*(xhx*vhx+xhy*vhy+xhz*vhz) &
          + tmp*((spiny*xhz-spinz*xhy-vhx)*vhx &
                 +(spinz*xhx-spinx*xhz-vhy)*vhy &
                 +(spinx*xhy-spiny*xhx-vhz)*vhz)) &
              + m(1)/(m(1)+m(j))*(N_tid_px*spinx+N_tid_py*spiny+N_tid_pz*spinz)
      !-------------------------------------------------------------------------
      return
  end subroutine dEdt_tides


  !-----------------------------------------------------------------------------
  !-----------------------------------------------------------------------------
  !---------------------------  ROTATION  --------------------------------------
  !-----------------------------------------------------------------------------
  !-----------------------------------------------------------------------------

  ! Cpi in Msun.AU^5.day-2
  ! Frot_r in Msun.day-2
  ! Frot_os and Frot_op in Msun.AU.day-1

  !-----------------------------------------------------------------------------
  ! Radial part of the rotation flattening induced force
  subroutine F_rot_rad (nbod,m,xhx,xhy,xhz,spin,R_star5,k2_star,R_plan5 &
         ,k2_plan,j,Frot_r)

      use physical_constant
      implicit none
      ! Input/Output
      integer,intent(in) :: nbod,j
      real(double_precision),intent(in) :: R_star5,k2_star,R_plan5,k2_plan
      real(double_precision),intent(in) :: xhx,xhy,xhz
      real(double_precision),intent(in) :: spin(3,10)
      real(double_precision),intent(in) :: m(nbod)
      real(double_precision), intent(out) :: Frot_r
      ! Local
      real(double_precision) :: Cpi,Csi,rscalspinp,rscalspins
      real(double_precision) :: normspin_2p,normspin_2s,r_2,rr,r_4,r_5,r_7,r_8
      !-------------------------------------------------------------------------
      call rad_power (xhx,xhy,xhz,r_2,rr,r_4,r_5,r_7,r_8)
      call r_scal_spin (xhx,xhy,xhz,spin(1,j),spin(2,j),spin(3,j),rscalspinp)
      call r_scal_spin (xhx,xhy,xhz,spin(1,1),spin(2,1),spin(3,1),rscalspins)
      call norm_spin_2 (spin(1,j),spin(2,j),spin(3,j),normspin_2p)
      call norm_spin_2 (spin(1,1),spin(2,1),spin(3,1),normspin_2s)

      Cpi = m(1)*k2_plan*normspin_2p*R_plan5/(6.d0*K2)
      Csi = m(j)*k2_star*normspin_2s*R_star5/(6.d0*K2)

      Frot_r = -3.d0/r_5*(Csi+Cpi) &
           + 15.d0/r_7*(Csi*rscalspins*rscalspins/normspin_2s &
           +Cpi*rscalspinp*rscalspinp/normspin_2p)    
      !-------------------------------------------------------------------------
      return
  end subroutine F_rot_rad 

  !-----------------------------------------------------------------------------
  ! Orthoradial part of the rotation flattening of the star force
  subroutine F_rot_ortho_s (nbod,m,xhx,xhy,xhz,spinx,spiny,spinz,R_star5 &
         ,k2_star,j,Frot_os)

      use physical_constant
      implicit none
      ! Input/Output
      integer,intent(in) :: nbod,j
      real(double_precision),intent(in) :: R_star5,k2_star
      real(double_precision),intent(in) :: xhx,xhy,xhz
      real(double_precision),intent(in) :: spinx,spiny,spinz
      real(double_precision),intent(in) :: m(nbod)
      real(double_precision), intent(out) :: Frot_os
      ! Local
      real(double_precision) :: Cpi,Csi,rscalspins,normspin_2s
      real(double_precision) :: r_2,rr,r_4,r_5,r_7,r_8
      !-------------------------------------------------------------------------
      call rad_power (xhx,xhy,xhz,r_2,rr,r_4,r_5,r_7,r_8)
      call r_scal_spin (xhx,xhy,xhz,spinx,spiny,spinz,rscalspins)
      call norm_spin_2 (spinx,spiny,spinz,normspin_2s)

      Csi = m(j)*k2_star*normspin_2s*R_star5/(6.d0*K2)

      Frot_os =  -6.d0*Csi*rscalspins/(normspin_2s*r_5)  
      !-------------------------------------------------------------------------
      return
  end subroutine F_rot_ortho_s

  !-----------------------------------------------------------------------------
  ! Orthoradial part of the rotation flattening of the planet force
  subroutine F_rot_ortho_p (nbod,m,xhx,xhy,xhz,spinx,spiny,spinz,R_plan5 &
         ,k2_plan,j,Frot_op)

      use physical_constant
      implicit none
      ! Input/Output
      integer,intent(in) :: nbod,j
      real(double_precision),intent(in) :: R_plan5,k2_plan
      real(double_precision),intent(in) :: xhx,xhy,xhz
      real(double_precision),intent(in) :: spinx,spiny,spinz
      real(double_precision),intent(in) :: m(nbod)
      real(double_precision), intent(out) :: Frot_op
      ! Local
      real(double_precision) :: Cpi,Csi,rscalspinp,normspin_2p
      real(double_precision) :: r_2,rr,r_4,r_5,r_7,r_8
      !------------------------------------------------------------------------------
      call rad_power (xhx,xhy,xhz,r_2,rr,r_4,r_5,r_7,r_8)
      call r_scal_spin (xhx,xhy,xhz,spinx,spiny,spinz,rscalspinp)
      call norm_spin_2 (spinx,spiny,spinz,normspin_2p)

      Cpi = m(1)*k2_plan*normspin_2p*R_plan5/(6.d0*K2)

      Frot_op =  -6.d0*Cpi*rscalspinp/(normspin_2p*r_5)  
      !------------------------------------------------------------------------------
      return
  end subroutine F_rot_ortho_p

  !-----------------------------------------------------------------------------
  ! Torque acting on the planet
  subroutine Torque_rot_p (nbod,m,xhx,xhy,xhz,spinx,spiny,spinz &
       ,R_plan5,k2_plan,j,N_rot_px,N_rot_py,N_rot_pz)
  
      implicit none
      ! Input/Output
      integer,intent(in) :: nbod,j
      real(double_precision),intent(in) :: xhx,xhy,xhz
      real(double_precision),intent(in) :: spinx,spiny,spinz
      real(double_precision),intent(in) :: R_plan5,k2_plan
      real(double_precision),intent(in) :: m(nbod)
      real(double_precision), intent(out) :: N_rot_px,N_rot_py,N_rot_pz
      ! Local
      real(double_precision) :: Frot_op
      !-------------------------------------------------------------------------
      call F_rot_ortho_p (nbod,m,xhx,xhy,xhz,spinx,spiny,spinz,R_plan5 &
                          ,k2_plan,j,Frot_op)               

      N_rot_px = Frot_op*(xhy*spinz-xhz*spiny)
      N_rot_py = Frot_op*(xhz*spinx-xhx*spinz)
      N_rot_pz = Frot_op*(xhx*spiny-xhy*spinx) 
      !-------------------------------------------------------------------------
      return
  end subroutine Torque_rot_p  

  !-----------------------------------------------------------------------------
  ! Torque acting on the star
  subroutine Torque_rot_s (nbod,m,xhx,xhy,xhz,spinx,spiny,spinz &
       ,R_star5,k2_star,j,N_rot_sx,N_rot_sy,N_rot_sz)
  
      implicit none
      ! Input/Output
      integer,intent(in) :: nbod,j
      real(double_precision),intent(in) :: xhx,xhy,xhz
      real(double_precision),intent(in) :: spinx,spiny,spinz
      real(double_precision),intent(in) :: R_star5,k2_star
      real(double_precision),intent(in) :: m(nbod)
      real(double_precision), intent(out) :: N_rot_sx,N_rot_sy,N_rot_sz
      ! Local
      real(double_precision) :: Frot_os
      !-------------------------------------------------------------------------
      call F_rot_ortho_s (nbod,m,xhx,xhy,xhz,spinx,spiny,spinz,R_star5 &
                          ,k2_star,j,Frot_os)               

      N_rot_sx = Frot_os*(xhy*spinz-xhz*spiny)
      N_rot_sy = Frot_os*(xhz*spinx-xhx*spinz)
      N_rot_sz = Frot_os*(xhx*spiny-xhy*spinx) 
      !-------------------------------------------------------------------------
      return
  end subroutine Torque_rot_s  

  !-----------------------------------------------------------------------------
  ! Force due to the rotational flattening 
  subroutine F_rotation (nbod,m,xhx,xhy,xhz,spin,R_star5,k2_star,R_plan5 &
         ,k2_plan,j,F_rot_tot_x,F_rot_tot_y,F_rot_tot_z)

      use physical_constant
      implicit none
      ! Input/Output
      integer,intent(in) :: nbod,j
      real(double_precision),intent(in) :: xhx,xhy,xhz
      real(double_precision),intent(in) :: R_star5,k2_star
      real(double_precision),intent(in) :: R_plan5,k2_plan
      real(double_precision),intent(in) :: m(nbod),spin(3,10)
      real(double_precision), intent(out) :: F_rot_tot_x,F_rot_tot_y,F_rot_tot_z
      ! Local
      real(double_precision) :: Frot_r,Frot_os,Frot_op
      !-------------------------------------------------------------------------
      call F_rot_rad (nbod,m,xhx,xhy,xhz,spin,R_star5,k2_star,R_plan5,k2_plan &
                      ,j,Frot_r)
      call F_rot_ortho_s (nbod,m,xhx,xhy,xhz,spin(1,1),spin(2,1),spin(3,1) &
                      ,R_star5,k2_star,j,Frot_os)
      call F_rot_ortho_p (nbod,m,xhx,xhy,xhz,spin(1,j),spin(2,j),spin(3,j) &
                      ,R_plan5,k2_plan,j,Frot_op)

      F_rot_tot_x = Frot_r*xhx + Frot_op*spin(1,j) + Frot_os*spin(1,1)
      F_rot_tot_y = Frot_r*xhy + Frot_op*spin(2,j) + Frot_os*spin(2,1)
      F_rot_tot_z = Frot_r*xhz + Frot_op*spin(3,j) + Frot_os*spin(3,1)
      !-------------------------------------------------------------------------
      return
  end subroutine F_rotation


  !-----------------------------------------------------------------------------
  !-----------------------------------------------------------------------------
  !---------------------------   GENERAL  --------------------------------------
  !---------------------------  RELATIVITY  ------------------------------------
  !-----------------------------------------------------------------------------
  !-----------------------------------------------------------------------------

  ! FGR_rad in AU.day-2 and FGR_ort in day-1

  !-----------------------------------------------------------------------------
  ! Radial part of the GR force (Kidder 1995, Mardling & Lin 2002)
  subroutine F_GR_rad (nbod,m,xhx,xhy,xhz,vhx,vhy,vhz,GRparam,C2,j,FGR_rad)

      implicit none
      ! Input/Output
      integer,intent(in) :: nbod,j
      real(double_precision),intent(in) :: xhx,xhy,xhz,vhx,vhy,vhz
      real(double_precision),intent(in) :: GRparam,C2
      real(double_precision),intent(in) :: m(nbod)
      real(double_precision), intent(out) :: FGR_rad
      ! Local
      real(double_precision) :: r_2,rr,r_4,r_5,r_7,r_8,v_2,norm_v,v_rad
      !-------------------------------------------------------------------------
      call rad_power (xhx,xhy,xhz,r_2,rr,r_4,r_5,r_7,r_8)
      call velocities (xhx,xhy,xhz,vhx,vhy,vhz,v_2,norm_v,v_rad)

      FGR_rad = -(m(1)+m(j))/(r_2*C2*C2) &
           *((1.0d0+3.0d0*GRparam)*v_2 &  
           -2.d0*(2.d0+GRparam)*(m(1)+m(j))/rr &
           -1.5d0*GRparam*v_rad*v_rad) 
      !-------------------------------------------------------------------------
      return
  end subroutine F_GR_rad

  !-----------------------------------------------------------------------------
  ! Orthoradial part of the GR force
  subroutine F_GR_ortho (nbod,m,xhx,xhy,xhz,vhx,vhy,vhz,GRparam,C2,j,FGR_ort)

      implicit none
      ! Input/Output
      integer,intent(in) :: nbod,j
      real(double_precision),intent(in) :: xhx,xhy,xhz,vhx,vhy,vhz
      real(double_precision),intent(in) :: GRparam,C2
      real(double_precision),intent(in) :: m(nbod)
      real(double_precision), intent(out) :: FGR_ort
      ! Local
      real(double_precision) :: r_2,rr,r_4,r_5,r_7,r_8,v_2,norm_v,v_rad
      !-------------------------------------------------------------------------
      call rad_power (xhx,xhy,xhz,r_2,rr,r_4,r_5,r_7,r_8)
      call velocities (xhx,xhy,xhz,vhx,vhy,vhz,v_2,norm_v,v_rad)

      FGR_ort = (m(1)+m(j))/(r_2*C2*C2) &
                    *2.0d0*(2.0d0-GRparam)*v_rad*norm_v 
      !-------------------------------------------------------------------------
      return
  end subroutine F_GR_ortho

  !-----------------------------------------------------------------------------
  ! GR force
  subroutine F_GenRel (nbod,m,xhx,xhy,xhz,vhx,vhy,vhz,GRparam,C2,j &
         ,F_GR_tot_x,F_GR_tot_y,F_GR_tot_z)

      implicit none
      ! Input/Output
      integer,intent(in) :: nbod,j
      real(double_precision),intent(in) :: xhx,xhy,xhz,vhx,vhy,vhz
      real(double_precision),intent(in) :: GRparam,C2
      real(double_precision),intent(in) :: m(nbod)
      real(double_precision), intent(out) :: F_GR_tot_x,F_GR_tot_y,F_GR_tot_z
      ! Local
      real(double_precision) :: tmp,tmp1,FGR_rad,FGR_ort
      !-------------------------------------------------------------------------
      call F_GR_rad (nbod,m,xhx,xhy,xhz,vhx,vhy,vhz,GRparam,C2,j,FGR_rad)
      call F_GR_ortho (nbod,m,xhx,xhy,xhz,vhx,vhy,vhz,GRparam,C2,j,FGR_ort)

      tmp  = sqrt(xhx*xhx+xhy*xhy+xhz*xhz)
      tmp1 = sqrt(vhx*vhx+vhy*vhy+vhz*vhz)

      F_GR_tot_x = m(j)*(FGR_rad*xhx/tmp+FGR_ort*vhx/tmp1)
      F_GR_tot_y = m(j)*(FGR_rad*xhy/tmp+FGR_ort*vhy/tmp1)
      F_GR_tot_z = m(j)*(FGR_rad*xhz/tmp+FGR_ort*vhz/tmp1)
      !-------------------------------------------------------------------------
      return
  end subroutine F_GenRel

end module user_module
