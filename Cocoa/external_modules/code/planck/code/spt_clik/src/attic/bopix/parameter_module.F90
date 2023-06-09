   MODULE PARAMETER_MODULE
    
      IMPLICIT NONE

      PUBLIC

      INTEGER(4)                     ::   IERR
      REAL(8), PARAMETER             ::   ZERO = 0.0D0, ONE = 1.0D0, MONE = -1.0D0

      INTEGER(4),PARAMETER           ::   NSIDEOUT=16, BOPIX_LMAX_COV=64
      INTEGER(4)                     ::   NPIX
      INTEGER(4)                     ::   BOPIX_CL_FILE, BOPIX_CL_LMAX
      INTEGER(4), PARAMETER          ::   BOPIX_LMIN = 2, KMIN = 0 , KMAX = 5

      INTEGER(4)              ::   LDMAT_T, NMAT_T, LDNMAT_T, LDMAT_TP, NMAT_TP, LDNMAT_TP,&
                                   LDMAT_PT, NMAT_PT, LDNMAT_PT, LDMAT_P, NMAT_P, LDNMAT_P,&
                                   LDTREMAT, NTREMAT, LDNTREMAT, NPIX_T, NPIX_P, N

      INTEGER (4)             ::   BOPIX_NTHREADS, OMP_GET_NUM_THREADS, BOPIX_THREADS

      REAL(8)                 :: MENO_LOGLIK_FACTOR, TD_FACT, UNIT_FACT, UNIT_FACT_COV,TDFACT_COVNOISE
      REAL(8)                 :: FWHM_ARC, FWHM_DEG

      REAL(8),ALLOCATABLE,DIMENSION(:,:)       :: BEAMW, BEAMWQ, SBEAM, BOPIX_CL
      INTEGER(4),ALLOCATABLE,DIMENSION(:)      :: NPIXVECT_T, NPIXVECT_P, NPIXVECT_TOT
      REAL(8),ALLOCATABLE,DIMENSION(:,:)       :: MAPOUT  
      REAL(8),ALLOCATABLE,DIMENSION(:,:)       :: MASK
      REAL(8),ALLOCATABLE,DIMENSION(:,:)       :: COVSIG,COVNOISE
      INTEGER(4),ALLOCATABLE,DIMENSION(:,:)    :: PIXEL_INDEX_T, PIXEL_INDEX_TP, PIXEL_INDEX_P
      REAL(8),ALLOCATABLE,DIMENSION(:)         :: MAPOUT_T, MAPOUT_Q, MAPOUT_U, MAPOUT_ALL, CONTR_RES
      REAL(8),ALLOCATABLE,DIMENSION(:)         :: LL_VECTOR

      INTEGER(8)              :: SIZE_T0, SIZE_T1, SIZE_T2
      INTEGER(8)              :: SIZE_TP0, SIZE_TP1, SIZE_TP2
      INTEGER(8)              :: SIZE_P0, SIZE_P1, SIZE_P2

      INTEGER(8),ALLOCATABLE,DIMENSION(:,:) :: INDEX_T0, INDEX_T1, INDEX_T2
      INTEGER(8),ALLOCATABLE,DIMENSION(:,:) :: INDEX_TP0, INDEX_TP1, INDEX_TP2
      INTEGER(8),ALLOCATABLE,DIMENSION(:,:) :: INDEX_P0, INDEX_P1, INDEX_P2

      REAL(8),ALLOCATABLE,DIMENSION(:) :: PIJ_T_0, PIJ_T_1, PIJ_T_2
      REAL(8),ALLOCATABLE,DIMENSION(:) :: PIJ_TP_0, PIJ_TP_1, PIJ_TP_2
      REAL(8),ALLOCATABLE,DIMENSION(:) :: PIJ_P_0, PIJ_P_1, PIJ_P_2
      REAL(8),ALLOCATABLE,DIMENSION(:) :: COSGAMMA_TP_1,SINGAMMA_TP_1
      REAL(8),ALLOCATABLE,DIMENSION(:) :: COSG_SINA_P_1,COSG_COSA_P_1,SING_SINA_P_1,SING_COSA_P_1
      REAL(8),ALLOCATABLE,DIMENSION(:) :: COSG_SINA_P_2,COSG_COSA_P_2,SING_SINA_P_2,SING_COSA_P_2

      INTEGER(4)              :: INFO
      INTEGER(4)              :: ORDERING_COV_SIGNAL, CONVERT_ORD_MAP, CONVERT_ORD_MASK,&
                                     NFILES_MASK, MAP_TYPE, NOISE
      LOGICAL                 :: MAPMASK, FOUND


      INTEGER (4)             :: F_HAND, N_REC
#ifndef GFORTRAN
      REAL(4)   :: ETIME
#endif
      REAL(4)   :: TOT_WALL_TIME
      REAL(4),DIMENSION(2) :: ELAPSED

      CHARACTER (LEN=160) :: WINDOW_FILE, COV_FILE, MASKFILE, MASKFILE_T, MASKFILE_P 
      CHARACTER (LEN=160) :: KEY  
      CHARACTER (LEN=160) :: BOPIX_FILENAME, BOPIX_CL_FILENAME
      CHARACTER (LEN=160) :: MASKED_MAP_T, MASKED_MAP_P, MAP_FILE 
      CHARACTER (LEN=160) :: DEFS_FILE, BOPIX_DATA_DIR


#ifndef DFORTRAN
      EXTERNAL ETIME
#endif
      EXTERNAL OMP_SET_NUM_THREADS, OMP_GET_NUM_THREADS


   CONTAINS

!==================================================================================================================================

   SUBROUTINE BOPIX_PARAMETER_INIT

      USE HEALPIX_TYPES
      USE PIX_TOOLS
      USE ALM_TOOLS
!      USE UDGRADE_NR
      USE FITSTOOLS, ONLY: INPUT_MAP
      USE RNGMOD
      USE SIMPLE_PARSER
     
     IMPLICIT NONE
     
     INTEGER(4) :: INDEX1, INDEX2, I, L, J, IPIX, JPIX, K
     INTEGER(8) :: THIS_RECL
     REAL(8),PARAMETER             :: BOPIX_PI=3.14159265358979323846264338328D0
!     CHARACTER(LEN=120),OPTIONAL,INTENT(IN) :: BOPIX_INPUT_FILENAME

      BOPIX_THREADS=OMP_GET_NUM_THREADS()

#ifndef GFORTRAN
      WRITE(6,*)'BOPIX INITIALIZATION...', BOPIX_THREADS, '  THREADS'
      TOT_WALL_TIME = ETIME(ELAPSED)      
      WRITE(6,*)'CPU TIME',TOT_WALL_TIME, 'USER ', ELAPSED(1), 'SYSTEM ', ELAPSED(2),'ELAPSED TIME ', TOT_WALL_TIME/BOPIX_THREADS
#endif

!     IF (PRESENT(BOPIX_INPUT_FILENAME)) THEN
!      CALL READ_PARAMETER(BOPIX_INPUT_FILENAME)
!     ELSE
      CALL READ_PARAMETER
!     END IF
      
      NPIX = 12*NSIDEOUT**2 
      ALLOCATE(MASK(0:NPIX-1,1:3))

   SELECT CASE (NFILES_MASK)
    CASE (1)
        CALL INPUT_MAP(MASKFILE,MASK(0:NPIX-1,1:3),NPIX,3)
        IF (CONVERT_ORD_MASK .EQ. 1) CALL CONVERT_NEST2RING(NSIDEOUT,MASK(0:NPIX-1,1:3))
        IF (CONVERT_ORD_MASK .EQ. 2) CALL CONVERT_RING2NEST(NSIDEOUT,MASK(0:NPIX-1,1:3))
    CASE (2)
        CALL INPUT_MAP(MASKFILE_T,MASK(0:NPIX-1,1:1),NPIX,1)
        CALL INPUT_MAP(MASKFILE_P,MASK(0:NPIX-1,2:2),NPIX,1)
        CALL INPUT_MAP(MASKFILE_P,MASK(0:NPIX-1,3:3),NPIX,1)
        IF (CONVERT_ORD_MASK .EQ. 1)  CALL CONVERT_NEST2RING(NSIDEOUT,MASK(0:NPIX-1,1:3))
        IF (CONVERT_ORD_MASK .EQ. 2)  CALL CONVERT_RING2NEST(NSIDEOUT,MASK(0:NPIX-1,1:3))
     CASE DEFAULT
       WRITE(6,*)'FULL SKY'
        MASK(0:NPIX-1,1:3) = 1
     END SELECT


     NPIX_T  = 0
     NPIX_P  = 0	
     
   DO I = 0,NPIX-1
    IF (MASK(I,1) .GT. 0.D0) NPIX_T  = NPIX_T +1
    IF (MASK(I,2) .GT. 0.D0) NPIX_P  = NPIX_P +1
   END DO

   ALLOCATE(NPIXVECT_T(1:NPIX_T))
   ALLOCATE(NPIXVECT_P(1:NPIX_P))
   INDEX1 = 0
   INDEX2 = 0
    
   DO I = 0,NPIX-1
    IF (MASK(I,1) .GT. 0.D0)  THEN
         INDEX1 = INDEX1 +1
         NPIXVECT_T(INDEX1) = I+1
    END IF

    IF (MASK(I,2) .GT. 0.D0)  THEN
         INDEX2 = INDEX2 +1
         NPIXVECT_P(INDEX2) = I+1        
    END IF
   END DO
     N = NPIX_T + 2*NPIX_P

!  write(6,*) 'nmatrix sizes... ', N, npix_t, npix_p

      LDMAT_T  = NPIX_T
      NMAT_T   = NPIX_T
      LDNMAT_T = LDMAT_T*NMAT_T

      LDMAT_P  = NPIX_P
      NMAT_P   = NPIX_P
      LDNMAT_P = LDMAT_P*NMAT_P

      LDMAT_TP  = NPIX_T
      NMAT_TP   = NPIX_P
      LDNMAT_TP = LDMAT_TP*NMAT_TP

      LDMAT_PT  = NPIX_P
      NMAT_PT   = NPIX_T
      LDNMAT_PT = LDMAT_PT*NMAT_PT
      
      LDTREMAT  = N
      NTREMAT   = N
      LDNTREMAT = LDTREMAT*NTREMAT

   ALLOCATE(LL_VECTOR(1:BOPIX_LMAX_COV), STAT=INFO)
   ALLOCATE(COVSIG(N,N),STAT=INFO)
   ALLOCATE(COVNOISE(N,N),STAT=INFO)


   SIZE_T0 = 0 
   SIZE_T1 = 0  
   SIZE_T2 = 0
   SIZE_TP0 = 0 
   SIZE_TP1 = 0  
   SIZE_TP2 = 0
   SIZE_P0 = 0 
   SIZE_P1 = 0  
   SIZE_P2 = 0

   CALL CALC_SIZE

   write(6,*) SIZE_T0, SIZE_T1, SIZE_T2, (SIZE_T0 + SIZE_T1 + SIZE_T2 + SIZE_T1 + SIZE_T2), npix_t**2
   write(6,*) SIZE_TP0, SIZE_TP1, SIZE_TP2, (SIZE_TP0 + SIZE_TP1 + SIZE_TP2), npix_t*npix_p
   write(6,*) SIZE_P0, SIZE_P1, SIZE_P2, (SIZE_P0 + SIZE_P1 + SIZE_P2), npix_p**2

   ALLOCATE(INDEX_T0(1:SIZE_T0,1:2),STAT=INFO)
   ALLOCATE(INDEX_T1(1:SIZE_T1,1:2),STAT=INFO)
   ALLOCATE(INDEX_T2(1:SIZE_T2,1:2),STAT=INFO)

   ALLOCATE(INDEX_TP0(1:SIZE_TP0,1:2),STAT=INFO)
   ALLOCATE(INDEX_TP1(1:SIZE_TP1,1:2),STAT=INFO)
   ALLOCATE(INDEX_TP2(1:SIZE_TP2,1:2),STAT=INFO)

   ALLOCATE(INDEX_P0(1:SIZE_P0,1:2),STAT=INFO)
   ALLOCATE(INDEX_P1(1:SIZE_P1,1:2),STAT=INFO)
   ALLOCATE(INDEX_P2(1:SIZE_P2,1:2),STAT=INFO)

   ALLOCATE(PIJ_T_1(1:SIZE_T1),STAT=INFO)
   ALLOCATE(PIJ_T_2(1:SIZE_T2),STAT=INFO)
   ALLOCATE(PIJ_TP_1(1:SIZE_TP1),STAT=INFO)
   ALLOCATE(PIJ_P_1(1:SIZE_P1),STAT=INFO)
   ALLOCATE(PIJ_P_2(1:SIZE_P2),STAT=INFO)

   ALLOCATE(COSGAMMA_TP_1(1:SIZE_TP1),STAT=INFO)
   ALLOCATE(SINGAMMA_TP_1(1:SIZE_TP1),STAT=INFO)

   ALLOCATE(COSG_SINA_P_1(1:SIZE_P1),STAT=INFO)
   ALLOCATE(COSG_COSA_P_1(1:SIZE_P1),STAT=INFO)
   ALLOCATE(SING_SINA_P_1(1:SIZE_P1),STAT=INFO)
   ALLOCATE(SING_COSA_P_1(1:SIZE_P1),STAT=INFO)

   ALLOCATE(COSG_SINA_P_2(1:SIZE_P2),STAT=INFO)
   ALLOCATE(COSG_COSA_P_2(1:SIZE_P2),STAT=INFO)
   ALLOCATE(SING_SINA_P_2(1:SIZE_P2),STAT=INFO)
   ALLOCATE(SING_COSA_P_2(1:SIZE_P2),STAT=INFO)

   ALLOCATE(MAPOUT_ALL(1:NPIX_T+2*NPIX_P),STAT=INFO)
   ALLOCATE(CONTR_RES(1:NPIX_T+2*NPIX_P),STAT=INFO)

  ALLOCATE(BEAMWQ(BOPIX_LMIN:BOPIX_LMAX_COV,0:5),STAT=INFO)

   ALLOCATE(BEAMW(0:BOPIX_LMAX_COV,1:3),STAT=INFO)
    BEAMW = 0.D0
     CALL PIXEL_WINDOW(BEAMW, WINDOWFILE=WINDOW_FILE)

      FWHM_ARC = FWHM_DEG*60.D0

      ALLOCATE(SBEAM(0:BOPIX_LMAX_COV,1:3))
      SBEAM = 0.D0

      CALL GAUSSBEAM(FWHM_ARC,BOPIX_LMAX_COV,SBEAM)

      BEAMWQ(:,:)=0.D0

      DO L=BOPIX_LMIN,BOPIX_LMAX_COV
        BEAMWQ(L,0)= BEAMW(L,1)*BEAMW(L,1)*SBEAM(L,1)*SBEAM(L,1)
        BEAMWQ(L,1)= BEAMW(L,2)*BEAMW(L,2)*SBEAM(L,2)*SBEAM(L,2)
        BEAMWQ(L,2)= BEAMW(L,3)*BEAMW(L,3)*SBEAM(L,3)*SBEAM(L,3)
        BEAMWQ(L,3)= BEAMW(L,1)*BEAMW(L,2)*SBEAM(L,1)*SBEAM(L,2)
        BEAMWQ(L,4)= BEAMW(L,1)*BEAMW(L,3)*SBEAM(L,1)*SBEAM(L,3)
        BEAMWQ(L,5)= BEAMW(L,2)*BEAMW(L,3)*SBEAM(L,2)*SBEAM(L,3)
      END DO

   DEALLOCATE(SBEAM)
   DEALLOCATE(BEAMW)

      DO L=1,BOPIX_LMAX_COV
       LL_VECTOR(L)=DBLE(L)
      END DO


      CALL CONST_CALC
     
 ALLOCATE(NPIXVECT_TOT(1:NPIX_T+2*NPIX_P), STAT=INFO)
    NPIXVECT_TOT(1:NPIX_T)=NPIXVECT_T(1:NPIX_T)
    NPIXVECT_TOT(NPIX_T+1:NPIX_T+NPIX_P)=NPIX+NPIXVECT_P(1:NPIX_P)
    NPIXVECT_TOT(NPIX_T+NPIX_P+1:NPIX_T+2*NPIX_P)=NPIX+NPIX+NPIXVECT_P(1:NPIX_P)
  
     SELECT CASE (NOISE)
        CASE (1)
          WRITE(6,*)'LETTURA NOISE FILE GIA MASCHERATO'
          INQUIRE(iolength=this_recl) COVNOISE
          OPEN(UNIT=3 , FILE=COV_FILE, FORM='unformatted', access='direct', recl=this_recl)
              READ(3,rec=1) COVNOISE
          CLOSE(3)
        CASE (2)
          WRITE(6,*)'LETTURA NOISE FILE FULL SKY'
          INQUIRE(iolength=this_recl) COVNOISE(1,1)
          OPEN(UNIT=3 , FILE=COV_FILE, FORM='unformatted', access='direct', recl=this_recl)
           DO J=1,NTREMAT
            DO I=1,LDTREMAT
              IPIX=NPIXVECT_TOT(I)
              JPIX=NPIXVECT_TOT(J)
              N_REC=(JPIX-1)*3*NPIX+IPIX
              READ(3,REC=N_REC)COVNOISE(I,J)
            END DO
           END DO
          CLOSE(3)
!          INQUIRE(iolength=this_recl) COVNOISE
!          OPEN(UNIT=3 , FILE='masked_noise.bin', FORM='unformatted', access='direct', recl=this_recl)
!           WRITE(3,rec=1)COVNOISE
!          CLOSE(3)
    END SELECT
            COVNOISE = UNIT_FACT_COV*UNIT_FACT_COV*TDFACT_COVNOISE*TDFACT_COVNOISE*COVNOISE
   DEALLOCATE(NPIXVECT_TOT)         
         CALL MAP_INPUT

   SELECT CASE (BOPIX_CL_FILE)
     CASE(1)
      ALLOCATE(BOPIX_CL(0:5,0:200),STAT=INFO)
      BOPIX_CL=0.D0
!       BOPIX_CL_FILENAME=BOPIX_DATA_DIR//BOPIX_CL_FILENAME
       OPEN(UNIT=11,FILE=BOPIX_CL_FILENAME,STATUS='OLD',FORM='FORMATTED',ACTION='READ')
        DO L=2,200
!       IF BOPIX_CL_FILE HAS ALSO COLUMN FOR MULTIPOLES DECOMMENT BELOW
          READ(11,*) K, BOPIX_CL(0,L), BOPIX_CL(2,L), BOPIX_CL(3,L), BOPIX_CL(1,L)
!          READ(11,*) BOPIX_CL(0,L), BOPIX_CL(2,L), BOPIX_CL(3,L), BOPIX_CL(1,L)
!       IF BOPIX_CL_FILE HAS BANDPOWERS DECOMMENT BELOW 
!          WRITE (6,*) 'TT l=2,35', BOPIX_CL(0,2), BOPIX_CL(0,35)
          BOPIX_CL(0,L)=BOPIX_CL(0,L)*2.D0*BOPIX_PI/L/(L+1)
          BOPIX_CL(1,L)=BOPIX_CL(1,L)*2.D0*BOPIX_PI/L/(L+1)
          BOPIX_CL(2,L)=BOPIX_CL(2,L)*2.D0*BOPIX_PI/L/(L+1)
          BOPIX_CL(3,L)=BOPIX_CL(3,L)*2.D0*BOPIX_PI/L/(L+1) 
        END DO
       CLOSE(11)
   END SELECT

#ifndef GFORTRAN
      WRITE(6,*)'BOPIX INITIALIZED..'
      TOT_WALL_TIME = ETIME(ELAPSED)          
      WRITE(6,*)'CPU TIME',TOT_WALL_TIME, 'USER ', ELAPSED(1), 'SYSTEM ', ELAPSED(2),'ELAPSED TIME ', TOT_WALL_TIME/BOPIX_THREADS
#endif

   END SUBROUTINE BOPIX_PARAMETER_INIT
!=================================================================================================================================

   SUBROUTINE MAP_INPUT()

      USE HEALPIX_TYPES
      USE PIX_TOOLS
      USE ALM_TOOLS
      USE FITSTOOLS, ONLY: INPUT_MAP

     IMPLICIT NONE
     INTEGER(4)    :: I, K, IPIX, MAP_ORDERING
     INTEGER(4)    :: THIS_RECL
     REAL(8),DIMENSION(:),ALLOCATABLE :: MAPOUT_TMP, MAPT

        
     REAL(8),dimension(0:NPIX-1) :: MASKT
     
     REAL(8),DIMENSION(1:2)     :: ZBOUNDS
     REAL(8),DIMENSION(0:3)     :: MULTIPOLES
     


!     CALL OMP_SET_NUM_THREADS(1)
                
               SELECT CASE (MAP_TYPE)
                 CASE (1)
                       ALLOCATE(MAPOUT(0:NPIX-1,1:3),STAT=INFO)
                         write(6,*)'info ',info
                       ALLOCATE(MAPOUT_T(1:NPIX_T),STAT=INFO)
                       ALLOCATE(MAPOUT_Q(1:NPIX_P),STAT=INFO)
                       ALLOCATE(MAPOUT_U(1:NPIX_P),STAT=INFO)
       
                        WRITE(6,*)'LETTURA MAPPA FULL SKY DA FILE FITS TQU'
			CALL INPUT_MAP(MAP_FILE,MAPOUT(0:NPIX-1,1:3),NPIX,3)
                        write(6,*)'ciao'
			IF (CONVERT_ORD_MAP .EQ. 1) THEN
			 WRITE(6,*)'CONVERSIONE NEST-RING'
				CALL CONVERT_NEST2RING(NSIDEOUT,MAPOUT(0:NPIX-1,1:3))
			END IF
                        IF (CONVERT_ORD_MAP .EQ. 2) THEN
                         WRITE(6,*)'CONVERSIONE RING-NEST'
                                CALL CONVERT_RING2NEST(NSIDEOUT,MAPOUT(0:NPIX-1,1:3))
                        END IF
                    CASE (2)
                       WRITE(6,*)'LETTURA DA FILE ASCII NON SUPPORTATA'
                    CASE (3)
                        WRITE(6,*)'LETTURA MAPPA DA 2 FILES MASCHERATI BINARI (SIGNAL + NOISE)'
                   ALLOCATE(MAPOUT_TMP(1:NPIX_T+2*NPIX_P))
                   INQUIRE(iolength=this_recl) MAPOUT_TMP(1:NPIX_T)
                     OPEN(UNIT=3 , FILE=MASKED_MAP_T, FORM='unformatted', access='direct', recl=this_recl)
                      READ(3,rec=1) MAPOUT_TMP(1:NPIX_T)
                     CLOSE(3)

                   INQUIRE(iolength=this_recl) MAPOUT_TMP(NPIX_T+1:NPIX_T+2*NPIX_P)
                     OPEN(UNIT=3 , FILE=MASKED_MAP_P, FORM='unformatted', access='direct', recl=this_recl)
                      READ(3,rec=1) MAPOUT_TMP(NPIX_T+1:NPIX_T+2*NPIX_P)
                     CLOSE(3)

		END SELECT

   SELECT CASE (MAP_TYPE)
    CASE (1)       

!RIMOZIONE MONOPOLO DIPOLO
                                
             ALLOCATE(MAPT(0:NPIX-1))
            MAPT(0:NPIX-1) = MAPOUT(0:NPIX-1,1)
            MASKT=MASK(:,1)
            ZBOUNDS(1)=-1.D0
            ZBOUNDS(2)= 1.D0
          !ordering 1 RING; 2 NEST
            MAP_ORDERING = 2
            call remove_dipole(NSIDEOUT,MAPT,MAP_ORDERING,2,multipoles,zbounds,-1.6375d30,MASKT)
               PRINT*,'monopole removed =',multipoles(0)
               PRINT*,'dipole removed =',multipoles(1),multipoles(2),multipoles(3)
            MAPOUT(0:NPIX-1,1)=MAPT(0:NPIX-1)
                   
             DEALLOCATE(MAPT)
                      


         DO IPIX=1,NPIX_T
          I = NPIXVECT_T(IPIX)
          MAPOUT_T(IPIX)=TD_FACT*UNIT_FACT*MAPOUT(I-1,1)
         END DO

         DO IPIX=1,NPIX_P
          I = NPIXVECT_P(IPIX)          
          MAPOUT_Q(IPIX)=TD_FACT*UNIT_FACT*MAPOUT(I-1,2)
          MAPOUT_U(IPIX)=TD_FACT*UNIT_FACT*MAPOUT(I-1,3)
         END DO
          MAPOUT_ALL(1:NPIX_T)=MAPOUT_T
          MAPOUT_ALL(NPIX_T+1:NPIX_T+NPIX_P)=MAPOUT_Q
          MAPOUT_ALL(NPIX_T+NPIX_P+1:NPIX_T+2*NPIX_P)=MAPOUT_U
!         DEALLOCATE(MAPOUT_U)
!         DEALLOCATE(MAPOUT_Q)
!         DEALLOCATE(MAPOUT_T)
!         DEALLOCATE(MAPOUT)
    CASE (3)
         DO IPIX=1,NPIX_T+2*NPIX_P
          MAPOUT_ALL = UNIT_FACT*TD_FACT*MAPOUT_TMP
         END DO
         DEALLOCATE(MAPOUT_TMP)
   END SELECT

   write(6,*)'maps read...'

  END SUBROUTINE MAP_INPUT
!==================================================================================================================================
   
   SUBROUTINE ANGLE(NSIDEOUT,IPIX,JPIX,PHI)

    USE HEALPIX_TYPES
    USE PIX_TOOLS

    IMPLICIT NONE

    INTEGER(4) :: NSIDEOUT,IPIX,JPIX,COUNT
    REAL(8),DIMENSION(1:3) :: VECI,VECJ,RIJ,RISTAR,VECN 
    REAL(8) :: PHI,NORMRIJ,NORMRISTAR

    VECN=(/0_DP,0_DP,1_DP/)    

       SELECT CASE (ORDERING_COV_SIGNAL)
         CASE (1)
           CALL PIX2VEC_RING(NSIDEOUT,IPIX,VECI)
           CALL PIX2VEC_RING(NSIDEOUT,JPIX,VECJ)
         CASE(0)
           CALL PIX2VEC_NEST(NSIDEOUT,IPIX,VECI)
           CALL PIX2VEC_NEST(NSIDEOUT,JPIX,VECJ)
         END SELECT

    COUNT = 0

10  CALL VECT_PROD(VECI,VECJ,RIJ)
    CALL VECT_PROD(VECI,VECN,RISTAR)
    IF (DOT_PRODUCT(RIJ,RIJ).LT.1.D-20) THEN
       PHI=0.0D0
       GOTO 20
    ELSE IF (RISTAR(1)**2+RISTAR(2)**2+RISTAR(3)**2.LT.1.D-10) THEN
       VECI(1) = VECI(1)+SQRT(PI/3_DP)/NSIDEOUT/100_DP
       VECI(2) = VECI(2)+SQRT(PI/3_DP)/NSIDEOUT/100_DP
       COUNT = COUNT+1
       WRITE(6,*)'PASSAGGIO',COUNT
       GOTO 10
    ELSE
       NORMRIJ = SQRT(RIJ(1)**2+RIJ(2)**2+RIJ(3)**2)
       RIJ(:) = RIJ(:)/NORMRIJ
       NORMRISTAR = SQRT(RISTAR(1)**2+RISTAR(2)**2+RISTAR(3)**2)
       RISTAR(:) = RISTAR(:)/NORMRISTAR   
       CALL ANGDIST(RIJ,RISTAR,PHI)
!!$ VEDI A8 DEL TEGMARK PRD 2001
       IF (RIJ(3).LT.0.0D0) THEN
          PHI = -ABS(PHI)
       ELSE
          PHI = ABS(PHI)
       END IF
    END IF
20  CONTINUE
  END SUBROUTINE ANGLE
!=================================================================================================================================

   SUBROUTINE CONST_CALC

     USE HEALPIX_TYPES
     USE PIX_TOOLS

     IMPLICIT NONE

     REAL(8)                     :: BOPIX_GAMMA, BOPIX_ALFA, TMP1
     REAL(8),DIMENSION(1:3)      :: VECI,VECJ
     INTEGER(4)                  :: INDI,INDJ,KINDEX
     INTEGER(8)                  :: I0,I1,I2, I, J, TMP


     I0=1
     I1=1
     I2=1

!!$OMP DO PRIVATE(I,INDI,INDJ,VECI,VECJ) SCHEDULE(STATIC)
   DO J = 1, NMAT_T
    DO I = 1, J
!    DO I = 1, LDMAT_T
      INDI = NPIXVECT_T(I)
      INDJ = NPIXVECT_T(J)
       SELECT CASE (ORDERING_COV_SIGNAL)
         CASE (1)
           CALL PIX2VEC_RING(NSIDEOUT,INDI-1,VECI)
           CALL PIX2VEC_RING(NSIDEOUT,INDJ-1,VECJ)
         CASE(0)
           CALL PIX2VEC_NEST(NSIDEOUT,INDI-1,VECI)
           CALL PIX2VEC_NEST(NSIDEOUT,INDJ-1,VECJ)
         END SELECT
       TMP1 = DOT_PRODUCT(VECI,VECJ)

        IF ( INDI .EQ. INDJ) THEN
             INDEX_T0(I0,1)=I
             INDEX_T0(I0,2)=J
             I0 = I0 + 1
        ELSE
         TMP = INT(TMP1 -1.D0 -1.D-10)
           IF ( TMP .EQ. -2) THEN
             PIJ_T_2(I2) = TMP1
             INDEX_T2(I2,1)=I
             INDEX_T2(I2,2)=J
             I2 = I2 + 1
           ELSE
             PIJ_T_1(I1) = TMP1
             INDEX_T1(I1,1)=I
             INDEX_T1(I1,2)=J
             I1 = I1 + 1
           END IF
        END IF

    END DO
   END DO
!!$OMP END DO

     I0=1
     I1=1
     I2=1

!!$OMP DO PRIVATE(I,INDI,INDJ,VECI,VECJ,BOPIX_GAMMA) SCHEDULE(STATIC)
   DO J = 1, NMAT_TP
    DO I = 1, LDMAT_TP
      INDI = NPIXVECT_T(I)
      INDJ = NPIXVECT_P(J)
       SELECT CASE (ORDERING_COV_SIGNAL)
         CASE (1)
           CALL PIX2VEC_RING(NSIDEOUT,INDI-1,VECI)
           CALL PIX2VEC_RING(NSIDEOUT,INDJ-1,VECJ)
         CASE(0)
           CALL PIX2VEC_NEST(NSIDEOUT,INDI-1,VECI)
           CALL PIX2VEC_NEST(NSIDEOUT,INDJ-1,VECJ)
         END SELECT
        TMP1 = DOT_PRODUCT(VECI,VECJ)

        IF ( INDI .EQ. INDJ) THEN
             INDEX_TP0(I0,1)=I
             INDEX_TP0(I0,2)=J
             I0=I0+1  
        ELSE
          TMP = INT(TMP1 -1.D0 -1.D-10)
           IF ( TMP .EQ. -2) THEN
             INDEX_TP2(I2,1)=I
             INDEX_TP2(I2,2)=J
             I2=I2+1  
           ELSE
             PIJ_TP_1(I1) = TMP1
             INDEX_TP1(I1,1)=I
             INDEX_TP1(I1,2)=J
               CALL ANGLE(NSIDEOUT,INDJ-1,INDI-1,BOPIX_GAMMA)
               COSGAMMA_TP_1(I1) = COS(2.D0*BOPIX_GAMMA)
               SINGAMMA_TP_1(I1) = SIN(2.D0*BOPIX_GAMMA)
             I1 = I1 + 1
            END IF
        END IF
   END DO
  END DO
!!$OMP END DO

     I0=1
     I1=1
     I2=1

!!$OMP DO PRIVATE(I,INDI,INDJ,VECI,VECJ,BOPIX_ALFA,BOPIX_GAMMA) SCHEDULE(STATIC)
    DO J = 1, NMAT_P
     DO I = 1, LDMAT_P
       INDI = NPIXVECT_P(I)
       INDJ = NPIXVECT_P(J)
       SELECT CASE (ORDERING_COV_SIGNAL)
         CASE (1)
           CALL PIX2VEC_RING(NSIDEOUT,INDI-1,VECI)
           CALL PIX2VEC_RING(NSIDEOUT,INDJ-1,VECJ)
         CASE(0)
           CALL PIX2VEC_NEST(NSIDEOUT,INDI-1,VECI)
           CALL PIX2VEC_NEST(NSIDEOUT,INDJ-1,VECJ)
         END SELECT
        TMP1 = DOT_PRODUCT(VECI,VECJ)        
        
        IF ( INDI .EQ. INDJ) THEN
             INDEX_P0(I0,1)=I
             INDEX_P0(I0,2)=J
             I0=I0+1  
        ELSE
         TMP = INT(TMP1 -1.D0 -1.D-10)
           IF ( TMP .EQ. -2) THEN
             PIJ_P_2(I2) = TMP1
             INDEX_P2(I2,1)=I
             INDEX_P2(I2,2)=J
             CALL ANGLE(NSIDEOUT,INDI-1,INDJ-1,BOPIX_ALFA)
             CALL ANGLE(NSIDEOUT,INDJ-1,INDI-1,BOPIX_GAMMA)
             COSG_SINA_P_2(I2)= COS(2.D0*BOPIX_GAMMA)*SIN(2.D0*BOPIX_ALFA)
             COSG_COSA_P_2(I2)= COS(2.D0*BOPIX_GAMMA)*COS(2.D0*BOPIX_ALFA)       
             SING_SINA_P_2(I2)= SIN(2.D0*BOPIX_GAMMA)*SIN(2.D0*BOPIX_ALFA)       
             SING_COSA_P_2(I2)= SIN(2.D0*BOPIX_GAMMA)*COS(2.D0*BOPIX_ALFA)       
             I2 = I2 + 1
           ELSE
             PIJ_P_1(I1) = TMP1
             INDEX_P1(I1,1)=I
             INDEX_P1(I1,2)=J
             CALL ANGLE(NSIDEOUT,INDI-1,INDJ-1,BOPIX_ALFA)
             CALL ANGLE(NSIDEOUT,INDJ-1,INDI-1,BOPIX_GAMMA)
             COSG_SINA_P_1(I1)= COS(2.D0*BOPIX_GAMMA)*SIN(2.D0*BOPIX_ALFA)
             COSG_COSA_P_1(I1)= COS(2.D0*BOPIX_GAMMA)*COS(2.D0*BOPIX_ALFA)       
             SING_SINA_P_1(I1)= SIN(2.D0*BOPIX_GAMMA)*SIN(2.D0*BOPIX_ALFA)       
             SING_COSA_P_1(I1)= SIN(2.D0*BOPIX_GAMMA)*COS(2.D0*BOPIX_ALFA)       
             I1 = I1 + 1
           END IF
        END IF

    END DO
   END DO
!!$OMP END DO

!!$OMP END PARALLEL
   END SUBROUTINE CONST_CALC
!=================================================================================================================================

   SUBROUTINE CALC_SIZE()

     USE HEALPIX_TYPES
     USE PIX_TOOLS

     IMPLICIT NONE

     REAL(8),DIMENSION(1:3)      :: VECI,VECJ
     REAL(8)                     :: TMP1
     INTEGER(8)                  :: I0,I1,I2
     INTEGER(4)                  :: INDI,INDJ,KINDEX, I, J, TMP

!!$OMP DO PRIVATE(I,INDI,INDJ,VECI,VECJ,TMP,TMP1) REDUCTION(+:SIZE_T0,SIZE_T1,SIZE_T2) SCHEDULE(STATIC)
   DO J = 1, NMAT_T
    DO I = 1, J
      INDI = NPIXVECT_T(I)
      INDJ = NPIXVECT_T(J)
       SELECT CASE (ORDERING_COV_SIGNAL)
         CASE (1)
           CALL PIX2VEC_RING(NSIDEOUT,INDI-1,VECI)
           CALL PIX2VEC_RING(NSIDEOUT,INDJ-1,VECJ)
         CASE(0)
           CALL PIX2VEC_NEST(NSIDEOUT,INDI-1,VECI)
           CALL PIX2VEC_NEST(NSIDEOUT,INDJ-1,VECJ)
         END SELECT
        TMP1 = DOT_PRODUCT(VECI,VECJ)

        IF ( INDI .EQ. INDJ) THEN
          SIZE_T0 = SIZE_T0 + 1
        ELSE
         TMP = INT(TMP1 -1.D0 -1.D-10)
           IF ( TMP .EQ. -2) THEN
          SIZE_T2 = SIZE_T2 + 1
           ELSE
          SIZE_T1 = SIZE_T1 + 1
           END IF
        END IF

    END DO
   END DO
!!$OMP END DO

!!$OMP DO PRIVATE(I,INDI,INDJ,VECI,VECJ,TMP, TMP1) REDUCTION(+:SIZE_TP0,SIZE_TP1,SIZE_TP2) SCHEDULE(STATIC)
   DO J = 1, NMAT_TP
    DO I = 1, LDMAT_TP
      INDI = NPIXVECT_T(I)
      INDJ = NPIXVECT_P(J)
       SELECT CASE (ORDERING_COV_SIGNAL)
         CASE (1)
           CALL PIX2VEC_RING(NSIDEOUT,INDI-1,VECI)
           CALL PIX2VEC_RING(NSIDEOUT,INDJ-1,VECJ)
         CASE(0)
           CALL PIX2VEC_NEST(NSIDEOUT,INDI-1,VECI)
           CALL PIX2VEC_NEST(NSIDEOUT,INDJ-1,VECJ)
         END SELECT
        TMP1 = DOT_PRODUCT(VECI,VECJ)
        IF ( INDI .EQ. INDJ) THEN
          SIZE_TP0 = SIZE_TP0 + 1
        ELSE
         TMP = INT(TMP1 -1.D0 -1.D-10)
           IF ( TMP .EQ. -2) THEN
          SIZE_TP2 = SIZE_TP2 + 1
           ELSE
          SIZE_TP1 = SIZE_TP1 + 1
           END IF
        END IF

   END DO
  END DO
!!$OMP END DO

!!$OMP DO PRIVATE(I,INDI,INDJ,VECI,VECJ,BOPIX_ALFA,BOPIX_GAMMA) REDUCTION(+:SIZE_P0,SIZE_P1,SIZE_P2) SCHEDULE(STATIC)
    DO J = 1, NMAT_P
     DO I = 1, LDMAT_P
       INDI = NPIXVECT_P(I)
       INDJ = NPIXVECT_P(J)
       SELECT CASE (ORDERING_COV_SIGNAL)
         CASE (1)
           CALL PIX2VEC_RING(NSIDEOUT,INDI-1,VECI)
           CALL PIX2VEC_RING(NSIDEOUT,INDJ-1,VECJ)
         CASE(0)
           CALL PIX2VEC_NEST(NSIDEOUT,INDI-1,VECI)
           CALL PIX2VEC_NEST(NSIDEOUT,INDJ-1,VECJ)
         END SELECT
        TMP1 = DOT_PRODUCT(VECI,VECJ)        
        
        IF ( INDI .EQ. INDJ) THEN
           SIZE_P0 = SIZE_P0 + 1
        ELSE
         TMP = INT(TMP1 -1.D0 -1.D-10)
           IF ( TMP .EQ. -2) THEN
            SIZE_P2 = SIZE_P2 + 1
           ELSE
            SIZE_P1 = SIZE_P1 + 1
           END IF
        END IF
    END DO
   END DO
!!$OMP END DO

!!$OMP END PARALLEL
   END SUBROUTINE CALC_SIZE

   END MODULE 
   
