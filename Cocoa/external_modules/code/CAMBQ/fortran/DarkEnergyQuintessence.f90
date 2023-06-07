    ! Equations module allowing for fairly general quintessence models
    !
    ! by Antony Lewis (http://cosmologist.info/)

    !!FIX March 2005: corrected update to next treatment of tight coupling
    !!Fix Oct 2011: a^2 factor in ayprime(EV%w_ix+1) [thanks Raphael Flauger]
    ! Oct 2013: update for latest CAMB, thanks Nelson Lima, Martina Schwind
    ! May 2020: updated for CAMB 1.x+

    ! Notes at http://antonylewis.com/notes/CAMB.pdf

    !This module is not well tested, use at your own risk!

    !Need to specify Vofphi function, and also initial_phi
    !You may also need to change other things to get it to work with different types of quintessence model

    !It works backwards, in that it assumes Omega_de is Omega_Q today, then does a binary search on the
    !initial conditions to find what is required to give that Omega_Q today after evolution.

    module Quintessence
    use DarkEnergyInterface
    use results
    use constants
    use classes
    implicit none
    private

    real(dl), parameter :: Tpl= sqrt(kappa*hbar/c**5)  ! sqrt(8 pi G hbar/c^5), reduced planck time

    ! General base class. Specific implemenetations should inherit, defining Vofphi and setting up
    ! initial conditions and interpolation tables
    type, extends(TDarkEnergyModel) :: TQuintessence
        integer :: DebugLevel = 0 !higher then zero for some debug output to console
        real(dl) :: astart = 1e-7_dl
        real(dl) :: integrate_tol = 1e-6_dl
        real(dl), dimension(:), allocatable :: sampled_a, phi_a, phidot_a
        ! Steps for log a and linear spacing, switching at max_a_log (set by Init)
        integer, private :: npoints_linear, npoints_log
        real(dl), private :: dloga, da, log_astart, max_a_log
        real(dl), private, dimension(:), allocatable :: ddphi_a, ddphidot_a
        class(CAMBdata), pointer, private :: State
    contains
    procedure :: Vofphi !V(phi) potential [+ any cosmological constant]
    procedure :: ValsAta !get phi and phi' at scale factor a, e.g. by interpolation in precomputed table
    procedure :: Init => TQuintessence_Init
    procedure :: PerturbedStressEnergy => TQuintessence_PerturbedStressEnergy
    procedure :: PerturbationEvolve => TQuintessence_PerturbationEvolve
    procedure :: BackgroundDensityAndPressure => TQuintessence_BackgroundDensityAndPressure
    procedure :: EvolveBackground
    procedure :: EvolveBackgroundLog
    procedure :: GetOmegaFromInitial ! DHFS
    procedure, private :: phidot_start => TQuintessence_phidot_start
    end type TQuintessence

    ! Specific implementation for late quintessence + cosmologial constant, assuming the early component
    ! energy density fraction is negligible at z=0.
    ! The specific parameterization of the potential implemented is the axion model of arXiv:1908.06995
    type, extends(TQuintessence) :: TLateQuintessence
        integer  :: which_potential
        real(dl) :: potentialparameter1, potentialparameter2 ! DFHS    
        real(dl) :: frac_lambda0 = 0._dl !fraction of dark energy density that is cosmological constant today
        integer  :: npoints = 5000 !baseline number of log a steps; will be increased if needed when there are oscillations
        real(dl), dimension(:), allocatable :: fde, ddfde

    contains
    procedure :: Vofphi => TLateQuintessence_VofPhi
    procedure :: Init => TLateQuintessence_Init
    procedure :: ReadParams =>  TLateQuintessence_ReadParams
    procedure, nopass :: PythonClass => TLateQuintessence_PythonClass
    procedure, nopass :: SelfPointer => TLateQuintessence_SelfPointer
    procedure, private :: check_error
    end type TLateQuintessence

    procedure(TClassDverk) :: dverk

    public TQuintessence, TLateQuintessence

    contains

    function VofPhi(this, phi, deriv)
    !Get the quintessence potential as function of phi
    !The input variable phi is sqrt(8*Pi*G)*psi, where psi is the field
    !Returns (8*Pi*G)^(1-deriv/2)*d^{deriv}V(psi)/d^{deriv}psi evaluated at psi
    !return result is in 1/Mpc^2 units [so times (Mpc/c)^2 to get units in 1/Mpc^2]
    class(TQuintessence) :: this
    real(dl) phi,Vofphi
    integer deriv

    call MpiStop('Quintessence classes must override to provide VofPhi')
    VofPhi = 0
    !if (deriv==0) then
    !    Vofphi= norm*this%m*exp(-this%sigma_model*phi)
    !else if (deriv ==1) then
    !    Vofphi=-norm*this%m*sigma_model*exp(-this%sigma_model*phi)
    !else if (deriv ==2) then
    !    Vofphi=norm*this%m*sigma_model**2*exp(-this%sigma_model*phi)
    !else
    !    stop 'Invalid deriv in Vofphi'
    !end if
    !VofPhi = VOfPhi* MPC_in_sec**2 /Tpl**2  !convert to units of 1/Mpc^2


    end function VofPhi


    subroutine TQuintessence_Init(this, State)
    class(TQuintessence), intent(inout) :: this
    class(TCAMBdata), intent(in), target :: State

    !Make interpolation table, etc,
    !At this point massive neutrinos have been initialized
    !so grho_no_de can be used to get density and pressure of other components at scale factor a

    select type(State)
    class is (CAMBdata)
        this%State => State
    end select

    this%is_cosmological_constant = .false.
    this%num_perturb_equations = 2

    this%log_astart = log(this%astart)

    end subroutine  TQuintessence_Init

    subroutine TQuintessence_BackgroundDensityAndPressure(this, grhov, a, grhov_t, w)
    !Get grhov_t = 8*pi*rho_de*a**2 and (optionally) equation of state at scale factor a
    class(TQuintessence), intent(inout) :: this
    real(dl), intent(in) :: grhov, a
    real(dl), intent(out) :: grhov_t
    real(dl), optional, intent(out) :: w
    real(dl) V, a2, grhov_lambda, phi, phidot

    if (this%is_cosmological_constant) then
        grhov_t = grhov * a * a
        if (present(w)) w = -1_dl
    elseif (a >= this%astart) then
        a2 = a**2
        call this%ValsAta(a,phi,phidot)
        V = this%Vofphi(phi,0)
        grhov_t = phidot**2/2 + a2*V
        if (present(w)) then
            w = (phidot**2/2 - a2*V)/grhov_t
        end if
    else
        grhov_t=0
        if (present(w)) w = -1
    end if

    end subroutine TQuintessence_BackgroundDensityAndPressure

    subroutine EvolveBackgroundLog(this,num,loga,y,yprime)
    ! Evolve the background equation in terms of loga.
    ! Variables are phi=y(1), a^2 phi' = y(2)
    ! Assume otherwise standard background components
    class(TQuintessence) :: this
    integer num
    real(dl) y(num),yprime(num)
    real(dl) loga, a

    a = exp(loga)
    call this%EvolveBackground(num, a, y, yprime)
    yprime = yprime*a

    end subroutine EvolveBackgroundLog

    subroutine EvolveBackground(this,num,a,y,yprime)
    ! Evolve the background equation in terms of a.
    ! Variables are phi=y(1), a^2 phi' = y(2)
    ! Assume otherwise standard background components
    class(TQuintessence) :: this
    integer num
    real(dl) y(num),yprime(num)
    real(dl) a, a2, tot
    real(dl) phi, grhode, phidot, adot

    a2=a**2
    phi = y(1)
    phidot = y(2)/a2

    grhode=a2*(0.5d0*phidot**2 + a2*this%Vofphi(phi,0))
    tot = this%state%grho_no_de(a) + grhode

    adot=sqrt(tot/3.0d0)
    yprime(1)=phidot/adot !d phi /d a
    yprime(2)= -a2**2*this%Vofphi(phi,1)/adot

    end subroutine EvolveBackground


    real(dl) function TQuintessence_phidot_start(this,phi)
    class(TQuintessence) :: this
    real(dl) :: phi

    TQuintessence_phidot_start = 0

    end function TQuintessence_phidot_start

    subroutine ValsAta(this,a,aphi,aphidot)
    class(TQuintessence) :: this
    !Do interpolation for background phi and phidot at a (precomputed in Init)
    real(dl) a, aphi, aphidot
    real(dl) a0,b0,ho2o6,delta,da
    integer ix

    if (a >= 0.9999999d0) then
        aphi= this%phi_a(this%npoints_linear+this%npoints_log)
        aphidot= this%phidot_a(this%npoints_linear+this%npoints_log)
        return
    elseif (a < this%astart) then
        aphi = this%phi_a(1)
        aphidot = 0
        return
    elseif (a > this%max_a_log) then
        delta= a-this%max_a_log
        ix = this%npoints_log + int(delta/this%da)
    else
        delta= log(a)-this%log_astart
        ix = int(delta/this%dloga)+1
    end if
    da = this%sampled_a(ix+1) - this%sampled_a(ix)
    a0 = (this%sampled_a(ix+1) - a)/da
    b0 = 1 - a0
    ho2o6 = da**2/6._dl
    aphi=b0*this%phi_a(ix+1) + a0*(this%phi_a(ix)-b0*((a0+1)*this%ddphi_a(ix)+(2-a0)*this%ddphi_a(ix+1))*ho2o6)
    aphidot=b0*this%phidot_a(ix+1) + a0*(this%phidot_a(ix)-b0*((a0+1)*this%ddphidot_a(ix)+(2-a0)*this%ddphidot_a(ix+1))*ho2o6)

    end subroutine ValsAta

    subroutine TQuintessence_PerturbedStressEnergy(this, dgrhoe, dgqe, &
        a, dgq, dgrho, grho, grhov_t, w, gpres_noDE, etak, adotoa, k, kf1, ay, ayprime, w_ix)
    !Get density perturbation and heat flux
    class(TQuintessence), intent(inout) :: this
    real(dl), intent(out) :: dgrhoe, dgqe
    real(dl), intent(in) ::  a, dgq, dgrho, grho, grhov_t, w, gpres_noDE, etak, adotoa, k, kf1
    real(dl), intent(in) :: ay(*)
    real(dl), intent(inout) :: ayprime(*)
    integer, intent(in) :: w_ix
    real(dl) phi, phidot, clxq, vq

    call this%ValsAta(a,phi,phidot)
    clxq=ay(w_ix)
    vq=ay(w_ix+1)
    dgrhoe= phidot*vq +clxq*a**2*this%Vofphi(phi,1)
    dgqe= k*phidot*clxq

    end subroutine TQuintessence_PerturbedStressEnergy


    subroutine TQuintessence_PerturbationEvolve(this, ayprime, w, w_ix, &
        a, adotoa, k, z, y)
    !Get conformal time derivatives of the density perturbation and velocity
    class(TQuintessence), intent(in) :: this
    real(dl), intent(inout) :: ayprime(:)
    real(dl), intent(in) :: a, adotoa, w, k, z, y(:)
    integer, intent(in) :: w_ix
    real(dl) clxq, vq, phi, phidot

    call this%ValsAta(a,phi,phidot) !wasting time calling this again..
    clxq=y(w_ix)
    vq=y(w_ix+1)
    ayprime(w_ix)= vq
    ayprime(w_ix+1) = - 2*adotoa*vq - k*z*phidot - k**2*clxq - a**2*clxq*this%Vofphi(phi,2)

    end subroutine TQuintessence_PerturbationEvolve

    ! Early Quintessence example, axion potential from e.g. arXiv: 1908.06995

    function TLateQuintessence_VofPhi(this, phi, deriv) result(VofPhi)
    !The input variable phi is sqrt(8*Pi*G)*psi
    !Returns (8*Pi*G)^(1-deriv/2)*d^{deriv}V(psi)/d^{deriv}psi evaluated at psi
    !return result is in 1/Mpc^2 units [so times (Mpc/c)^2 to get units in 1/Mpc^2]
    class(TLateQuintessence) :: this
    real(dl) phi,Vofphi
    integer deriv
    real(dl), parameter :: units = MPC_in_sec**2 /Tpl**2  !convert to units of 1/Mpc^2
    real(dl) :: V0, alpha 
    
    this%potentialparameter2 = this%state%CP%potentialparameter2
    this%which_potential = this%state%CP%which_potential
 
    ! DHFS: Begins potentials
    select case(this%state%CP%which_potential) 
        case(1) ! DHFS: Model 1: Exponential potential
            V0    = this%potentialparameter1
            alpha = this%potentialparameter2
            if (deriv == 0) then
                Vofphi = units * V0 * exp(-alpha * phi)  
            else if (deriv == 1) then
                Vofphi = units * V0 * (-alpha) * exp(-alpha * phi)
            else if (deriv == 2) then
                Vofphi = units * V0 * alpha**2 * exp(-alpha * phi)
            end if

        case(2) ! DHFS: Model 2: Polinomial potential
            V0    = this%potentialparameter1
            alpha = this%potentialparameter2
            if (deriv == 0) then
                Vofphi = units * V0 * phi**alpha 
            else if (deriv == 1) then
                Vofphi = units * V0 * alpha * phi**(alpha - 1)
            else if (deriv == 2) then
                Vofphi = units * V0 * alpha * (alpha - 1) * phi**(alpha - 2)
            end if

        case(3) ! DHFS: Model 3: Hiperbolic cossine potential (arXiv 1810.08586)
            V0    = this%potentialparameter1
            alpha = this%potentialparameter2 
            if (deriv == 0) then
                Vofphi = units * V0 * cosh(alpha*phi) 
            else if (deriv == 1) then
                Vofphi = units * V0 * alpha * sinh(alpha*phi)
            else if (deriv == 2) then
                Vofphi = units * V0 * alpha**2 * cosh(alpha*phi)
            end if
        
        case(4) ! DHFS: Model 4: Peebles-Ratra potential (arXiv 2004.00610)
            V0    = this%potentialparameter1
            alpha = this%potentialparameter2
            if (deriv == 0) then
                Vofphi = units * V0 * phi**(-alpha) 
            else if (deriv == 1) then
                Vofphi = units * V0 * (-alpha) * phi**(-alpha - 1)
            else if (deriv == 2) then
                Vofphi = units * V0 * alpha * (alpha + 1) * phi**(-alpha - 2) 
            end if

        case(5) ! DHFS: ZWS: (arXiv 9807002)
            V0    = this%potentialparameter1
            alpha = this%potentialparameter2
            if (deriv == 0) then
                Vofphi = units * V0 * (exp(alpha/phi) - 1) 
            else if (deriv == 1) then
                Vofphi = -units * V0 * alpha * exp(alpha/phi)/phi**2
            else if (deriv == 2) then
                Vofphi = units * V0 * alpha * (alpha + 2*phi) * exp(alpha/phi)/phi**4 
            end if     

        case(6) ! DHFS: Model 6: SUGRA -supergravity-
                ! Ph. Brax and J. Martin- arXiv 9905040
                ! Planck DE and MG arXiv 1502.01590
            V0    = this%potentialparameter1
            alpha = this%potentialparameter2
            if (deriv == 0) then
                Vofphi = units * V0 * phi**(-alpha) * exp(phi**2)
            else if (deriv == 1) then
                Vofphi = units * V0  * (-alpha + 2*phi**2) * phi**(-alpha - 1) * exp(phi**2)
            else if (deriv == 2) then
                Vofphi = units * V0 * (alpha * (1 + alpha) + 2*phi**2 * (1 - 2*alpha + 2*phi**2)) &
                       * phi**(-alpha - 2) * exp(phi**2)
            end if 

        case(7) ! DHFS: Model 7: PNGB -pseudo Nambu Goldstone bosons-
                ! Frieman et al. arXiv 9505060;
                ! Kaloper & Sorbo arXiv 0511543;
                ! Planck DE and MG arXiv 1502.01590
            V0    = this%potentialparameter1
            alpha = this%potentialparameter2
            if (deriv == 0) then
                Vofphi = units * V0 * (1 + cos(alpha * phi)) 
            else if (deriv == 1) then
                Vofphi = -units * V0 * alpha * sin(alpha * phi) 
            else if (deriv == 2) then
                Vofphi = -units * V0 * alpha**2 * cos(alpha * phi) 
            end if 
        case default
            write(*,*)'[Error] The kind of potential does not exist.'                                         
    end select
    ! DHFS: Ends potential

    end function TLateQuintessence_VofPhi

   ! DHFS start: Binary search in V0
    subroutine BinarySearchInV0(this, V0_min, V0_max, initial_phi, atol)

        class(TLateQuintessence), intent(inout) :: this

        real(dl), intent(inout) :: V0_min, V0_max, initial_phi, atol
        real(dl), parameter :: omega_tol = 1e-3 !DHFS
        real(dl) :: V0_middle, delta_V0, om, om1, om2, initial_phidot, astart
        integer  :: iter
        logical  :: OK

        astart = this%astart
        initial_phidot =  astart*this%phidot_start(initial_phi)

        this%potentialparameter1 = V0_min 
        om1 = this%GetOmegaFromInitial(astart, initial_phi, initial_phidot, atol)
        this%potentialparameter1 = V0_max
        om2 = this%GetOmegaFromInitial(astart, initial_phi, initial_phidot, atol)

        ! om_min = min(om1, om2)
        ! om_max = max(om1, om2)

        write(*,*) 'Target omega_de:', this%state%omega_de
        write(*,*) 'First trial om1:', om1 !, V0_min, this%potentialparameter1
        write(*,*) 'First trial om2:', om2 !, V0_max, this%potentialparameter1

        if (abs(om1 - this%state%omega_de) > omega_tol) then
            !if not, do binary search in V0 interval 
            OK = .false.   
            if ((om1 < this%state%omega_de .and. om2 < this%state%omega_de) .or. &
                (om1 > this%state%omega_de .and. om2 > this%state%omega_de)) then
                write (*,*) '[error]: V0_min and V0_max values must bracket required value.'
                write (*,*) 'om1, om2 = ', real(om1), real(om2)
                stop
            end if
            do iter = 1, 200 !DHFS: number of times that we want to split the interval [V0_min, V0_max] in a half
                V0_middle = (V0_min + V0_max) / 2.0

                this%potentialparameter1 = V0_middle
                om = this%GetOmegaFromInitial(astart,initial_phi,initial_phidot,atol)
                
                check_bs: if (abs(om - this%state%omega_de) < 1d-4) then
                    OK = .true.
                    this%potentialparameter1 = V0_middle
                    if (FeedbackLevel > 0) write(*,*) '[successful]: V0 = ',this%potentialparameter1
                    exit
                end if check_bs
                   
                split_bs: if (om < this%state%omega_de) then
                    V0_min = V0_middle
                    ! write (*,*) 'i:',iter,'om1:',om,'V0_min:',V0_min
                else if (om > this%state%omega_de) then
                    V0_max = V0_middle
                    ! write (*,*) 'i:',iter,'om2:',om,'V0_min:',V0_max
                end if split_bs

            end do  
            if (.not. OK) stop 'Search for good initial conditions did not converge' !this shouldn't happen
                          !exception ao inves de stop !!!!
        end if     
    end subroutine BinarySearchInV0
    ! DHFS ends: Binary search in V0       

    !DHFS Begins: background outputs 
    ! subroutine Background_Outputs(this,State,ix_iter)
    ! ! Return energy density for radiation, matter and dark energy in GeV^4
    !     integer, intent(in) :: ix_iter
    !     real(dl) :: a2 = sampled_a(ix_ter)**2
    !     real(dl),intent(out) :: fcdm 
        
    !     fcdm = this%state%grho_cdm(sampled_a(ix_iter))/(this%state%grho_no_de(sampled_a(ix_iter)) & 
    !          + a2*(0.5d0* phidot_a(ix_iter)**2 + a2*this%Vofphi(y(1),0)))
    ! end subroutine Background_Outputs
! DHFS Ends: background outputs

    subroutine TLateQuintessence_Init(this, State)
    use Powell
    class(TLateQuintessence), intent(inout) :: this
    class(TCAMBdata), intent(in), target :: State
    real(dl) aend, afrom
    integer, parameter ::  NumEqs=2
    real(dl) c(24),w(NumEqs,9), y(NumEqs)
    integer ind, i, ix
    real(dl), parameter :: splZero = 0._dl
    real(dl) initial_phi, initial_phidot, a2
    real(dl), dimension(:), allocatable :: sampled_a, phi_a, phidot_a, fde
    integer npoints, tot_points
    integer iflag, iter
    Type(TTimer)  :: Timer
    Type(TNEWUOA) :: Minimize
    real(dl) log_params(2), param_min(2), param_max(2)

    ! DHFS: Begins- variables for binary search in V0
    logical OK 
    real(dl) :: atol, om1, om2, om, phi ,astart
    real(dl) :: V0_min, V0_max 
    ! DHFS: Ends- variables for binary search in V0

    ! DHFS: background variables
    real(dl) :: fcdm, fbaryon, fphoton, fmasslessNu, fmassiveNu, cambH

    !Make interpolation table, etc,
    !At this point massive neutrinos have been initialized
    !so grho_no_de can be used to get density and pressure of other components at scale factor a

    call this%TQuintessence%Init(State)

    this%dloga = (-this%log_astart)/(this%npoints-1)

    !use log spacing in a up to max_a_log, then linear. Switch where step matches
    this%max_a_log = 1.d0/this%npoints/(exp(this%dloga)-1)
    npoints = (log(this%max_a_log)-this%log_astart)/this%dloga + 1

    if (allocated(this%phi_a)) then
        deallocate(this%phi_a,this%phidot_a)
        deallocate(this%ddphi_a,this%ddphidot_a, this%sampled_a)
    end if
    allocate(phi_a(npoints),phidot_a(npoints), sampled_a(npoints), fde(npoints))

    astart = this%astart 
    atol   = 1d-20
    initial_phi = 1.0_dl

    !DHFS: Begins- binary search
    if (this%state%CP%want_binary_search) then
        V0_min = 1e-127_dl   !127 DHFS: Assumed to be in Mpl^4
        V0_max = 1e-96_dl   !110 DHFS: Assumed to be in Mpl^4
        call BinarySearchInV0(this, V0_min, V0_max, initial_phi, atol)
    else 
        this%potentialparameter1 = this%state%CP%potentialparameter1    
    end if
    !DHFS: Ends- binary search

    y(1)=initial_phi
    initial_phidot =  this%astart*this%phidot_start(initial_phi)
    y(2)= initial_phidot*this%astart**2

    phi_a(1)=y(1)
    phidot_a(1)=y(2)/this%astart**2
    sampled_a(1)=this%astart
    
    ind=1
    afrom=this%log_astart
    ! open(10, file='background_outputs/outs_test.txt') ! DHFS: open the file
    do i=1, npoints-1
        aend = this%log_astart + this%dloga*i
        ix = i+1
        sampled_a(ix)=exp(aend)
        a2 = sampled_a(ix)**2
        call dverk(this,NumEqs,EvolveBackgroundLog,afrom,y,aend,this%integrate_tol,ind,c,NumEqs,w)
        if (.not. this%check_error(exp(afrom), exp(aend))) return
        call EvolveBackgroundLog(this,NumEqs,aend,y,w(:,1))
        
        phi_a(ix)=y(1)
        phidot_a(ix)=y(2)/a2

        !Define fde as ratio of early dark energy density to total
        fde(ix) = 1/((this%state%grho_no_de(sampled_a(ix)) +  this%frac_lambda0*this%State%grhov*a2**2) &
                / (a2*(0.5d0* phidot_a(ix)**2 + a2*this%Vofphi(y(1),0))) + 1)
        
        ! if (this%state%CP%want_background_outputs) then 

        ! fcdm = this%state%grho_cdm(sampled_a(ix))/(this%state%grho_no_de(sampled_a(ix)) & 
        !      + a2*(0.5d0* phidot_a(ix)**2 + a2*this%Vofphi(y(1),0))) 

        ! fbaryon = this%state%grho_baryon(sampled_a(ix))/(this%state%grho_no_de(sampled_a(ix)) & 
        !         + a2*(0.5d0* phidot_a(ix)**2 + a2*this%Vofphi(y(1),0)))                 

        ! fphoton = this%state%grho_photon(sampled_a(ix))/(this%state%grho_no_de(sampled_a(ix)) & 
        !         + a2*(0.5d0* phidot_a(ix)**2 + a2*this%Vofphi(y(1),0)))

        ! fmasslessNu = this%state%grho_masslessNu(sampled_a(ix))/(this%state%grho_no_de(sampled_a(ix)) & 
        !             + a2*(0.5d0* phidot_a(ix)**2 + a2*this%Vofphi(y(1),0)))

        ! fmassiveNu = this%state%grho_massiveNu(sampled_a(ix))/(this%state%grho_no_de(sampled_a(ix)) & 
        !             + a2*(0.5d0* phidot_a(ix)**2 + a2*this%Vofphi(y(1),0)))   

        ! cambH = sqrt((this%state%grho_no_de(sampled_a(ix)) + a2*(0.5d0* phidot_a(ix)**2 + a2*this%Vofphi(y(1),0)))/3) &
        !         /sampled_a(ix)**2     
    
        ! write(10,*) sampled_a(ix), phi_a(ix), phidot_a(ix), fde(ix), fcdm, fbaryon, fphoton, fmasslessNu , fmassiveNu, cambH
        !  end if   

    end do

    ! Do remaining steps with linear spacing in a, trying to be small enough
    this%npoints_log = ix
    this%max_a_log = sampled_a(ix)
    this%da = this%max_a_log *(exp(this%dloga)-1) 
    this%npoints_linear = int((1- this%max_a_log)/ this%da)+1
    this%da = (1- this%max_a_log)/this%npoints_linear

    tot_points = this%npoints_log+this%npoints_linear
    allocate(this%phi_a(tot_points),this%phidot_a(tot_points))
    allocate(this%ddphi_a(tot_points),this%ddphidot_a(tot_points))
    allocate(this%sampled_a(tot_points), this%fde(tot_points), this%ddfde(tot_points))
    this%sampled_a(1:ix) = sampled_a(1:ix)
    this%phi_a(1:ix) = phi_a(1:ix)
    this%phidot_a(1:ix) = phidot_a(1:ix)
    this%sampled_a(1:ix) = sampled_a(1:ix)
    this%fde(1:ix) = fde(1:ix)

    ind=1
    afrom = this%max_a_log
    do i=1, this%npoints_linear
        ix = this%npoints_log + i
        aend = this%max_a_log + this%da*i
        a2 =aend**2
        this%sampled_a(ix)=aend
        call dverk(this,NumEqs,EvolveBackground,afrom,y,aend,this%integrate_tol,ind,c,NumEqs,w)
        if (.not. this%check_error(afrom, aend)) return
        call EvolveBackground(this,NumEqs,aend,y,w(:,1))

        this%phi_a(ix)=y(1)
        this%phidot_a(ix)=y(2)/a2

        this%fde(ix) = 1/((this%state%grho_no_de(aend) +  this%frac_lambda0*this%State%grhov*a2**2) &
                     / (a2*(0.5d0* this%phidot_a(ix)**2 + a2*this%Vofphi(y(1),0))) + 1)

        ! if (want_background_outputs) then

        ! fcdm = this%state%grho_cdm(this%sampled_a(ix))/(this%state%grho_no_de(this%sampled_a(ix)) & 
        !         + a2*(0.5d0* this%phidot_a(ix)**2 + a2*this%Vofphi(y(1),0))) 

        ! fbaryon = this%state%grho_baryon(this%sampled_a(ix))/(this%state%grho_no_de(this%sampled_a(ix)) & 
        !         + a2*(0.5d0* this%phidot_a(ix)**2 + a2*this%Vofphi(y(1),0)))                 

        ! fphoton = this%state%grho_photon(this%sampled_a(ix))/(this%state%grho_no_de(this%sampled_a(ix)) & 
        !         + a2*(0.5d0* this%phidot_a(ix)**2 + a2*this%Vofphi(y(1),0)))

        ! fmasslessNu = this%state%grho_masslessNu(this%sampled_a(ix))/(this%state%grho_no_de(this%sampled_a(ix)) & 
        !             + a2*(0.5d0* this%phidot_a(ix)**2 + a2*this%Vofphi(y(1),0)))

        ! fmassiveNu = this%state%grho_massiveNu(this%sampled_a(ix))/(this%state%grho_no_de(this%sampled_a(ix)) & 
        !             + a2*(0.5d0* this%phidot_a(ix)**2 + a2*this%Vofphi(y(1),0)))   

        ! cambH = sqrt((this%state%grho_no_de(this%sampled_a(ix)) + a2*(0.5d0* this%phidot_a(ix)**2 & 
        !         + a2*this%Vofphi(y(1),0)))/3) /this%sampled_a(ix)**2     
        
        ! write(10,*) this%sampled_a(ix), this%phi_a(ix), this%phidot_a(ix), this%fde(ix), fcdm, fbaryon, fphoton, fmasslessNu , fmassiveNu, cambH
        !  end if

    end do
    ! close(10) ! DHFS: close the file
    
    call spline(this%sampled_a,this%phi_a,tot_points,splZero,splZero,this%ddphi_a)
    call spline(this%sampled_a,this%phidot_a,tot_points,splZero,splZero,this%ddphidot_a)
    call spline(this%sampled_a,this%fde,tot_points,splZero,splZero,this%ddfde)

    write(*,*)'potential type', this%state%CP%which_potential
    write(*,*)'binary search decision:', this%state%CP%want_binary_search
    write(*,*)'parameter1 from inifile:', this%state%CP%potentialparameter1
    write(*,*)'parameter1 from binary search:', this%potentialparameter1
    ! write(*,*)'om_de',this%state%omega_de
    ! write(*,*)'om_phi',this%GetOmegaFromInitial(astart, initial_phi, initial_phidot, atol)

    end subroutine TLateQuintessence_Init

    logical function check_error(this, afrom, aend)
    class(TLateQuintessence) :: this
    real(dl) afrom, aend

    if (global_error_flag/=0) then
        write(*,*) 'TLateQuintessence error integrating', afrom, aend
        stop
        check_error = .false.
        return
    end if
    check_error= .true.
    end function check_error

    subroutine TLateQuintessence_ReadParams(this, Ini)
    use IniObjects
    class(TLateQuintessence) :: this
    class(TIniFile), intent(in) :: Ini

    call this%TDarkEnergyModel%ReadParams(Ini)

    end subroutine TLateQuintessence_ReadParams

    function TLateQuintessence_PythonClass()
    character(LEN=:), allocatable :: TLateQuintessence_PythonClass

    TLateQuintessence_PythonClass = 'LateQuintessence'

    end function TLateQuintessence_PythonClass

    subroutine TLateQuintessence_SelfPointer(cptr,P)
    use iso_c_binding
    Type(c_ptr) :: cptr
    Type (TLateQuintessence), pointer :: PType
    class (TPythonInterfacedClass), pointer :: P

    call c_f_pointer(cptr, PType)
    P => PType

    end subroutine TLateQuintessence_SelfPointer

    real(dl) function GetOmegaFromInitial(this,astart,phi,phidot,atol)
    !Get omega_de today given particular conditions phi and phidot at a = astart
    class(TQuintessence) :: this
    real(dl), intent(IN) :: astart, phi,phidot, atol
    integer, parameter ::  NumEqs=2
    real(dl) c(24),w(NumEqs,9), y(NumEqs), ast
    integer ind, i
    
    ast=astart
    ind=1
    y(1)=phi
    y(2)=phidot*astart**2
    call dverk(this,NumEqs,EvolveBackground,ast,y,1._dl,atol,ind,c,NumEqs,w)
    call EvolveBackground(this,NumEqs,1._dl,y,w(:,1))
    
    GetOmegaFromInitial=(0.5d0*y(2)**2 + this%Vofphi(y(1),0))/this%State%grhocrit !(3*adot**2)
    
    end function GetOmegaFromInitial

    end module Quintessence
