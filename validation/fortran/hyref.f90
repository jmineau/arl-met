! Reference reader for arlmet's real-file checks, built on HYSPLIT's own library
! (libhysplit.a: METINI/METSET, PAKINP, CXY2LL/CLL2XY/GBL2LL/GBL2XY, PRFSIG/PRFPRS/PRFECM).
!
! stdin line 1: directory (ending in /); line 2: file name; line 3: mode
!   grid    : then n, n lines "x y" (1-based grid units) -> lat lon; then m, m lines "lat lon" -> x y
!             writes grid_out.txt (GRID/DREC/FILE as METINI set them, then the conversions)
!   labels  : every record's 50-byte label read as metinp/xtrct_stn read it, and each INDX
!             record's extended header read as metset reads it -> labels_out.txt
!   values  : then n, n record numbers (1-based) -> values_out.bin (nx*ny REAL each, x fastest)
!             read with PAKINP as metinp calls it (full grid); values_hdr.txt per record
!   column  : then t (time step, 1-based in file order), n, n lines "i j" (1-based cells)
!             reads the time step's records at each cell with PAKINP (DIF records added to their
!             parent as metinp adds them), then runs the flag's profile routine (PRFSIG flag 1,
!             PRFPRS flag 2, PRFECM flag 4) with output every DZ m -> column_<k>.txt, raw_<k>.txt
PROGRAM hyref
  IMPLICIT NONE
  INCLUDE 'limits.inc'
  INCLUDE 'DEFGRID.INC'
  INCLUDE 'gblgrd.inc'
  INCLUDE 'DEFARG1.INC'
  INCLUDE 'tminit.inc'
  INCLUDE 'gbl2xy.inc'
  INCLUDE 'gbl2ll.inc'
  INCLUDE 'cll2xy.inc'
  INCLUDE 'cxy2ll.inc'
  INTERFACE
    SUBROUTINE PAKINP(RVAR,CVAR,NX,NY,NX1,NY1,LX,LY,PREC,NEXP,VAR1,KSUM)
      REAL,          INTENT(OUT)   :: rvar (:,:)
      CHARACTER(1),  INTENT(IN)    :: cvar (:)
      INTEGER,       INTENT(IN)    :: nx,ny,nx1,ny1,lx,ly
      REAL,          INTENT(IN)    :: prec
      INTEGER,       INTENT(IN)    :: nexp
      REAL,          INTENT(IN)    :: var1
      INTEGER,       INTENT(INOUT) :: ksum
    END SUBROUTINE PAKINP
  END INTERFACE

  CHARACTER(16)  :: mode
  REAL           :: plat, plon, xp, yp, clat, clon
  LOGICAL        :: back, wvert
  INTEGER        :: iconvect, ngrd, ntim, kvel, kyr, kmo, kda, khr, kmn
  INTEGER        :: n, k, l, m, nx, ny, lrec

  READ(*,'(A)') FILE(1,1)%DIR
  FILE(1,1)%DIR = ADJUSTL(FILE(1,1)%DIR)
  READ(*,'(A)') FILE(1,1)%METEO
  READ(*,'(A)') mode
  OPEN(70, FILE='MESSAGE')
  OPEN(85, FILE='WARNING')
  CALL TMINIT
  ngrd = 1; ntim = 1; kvel = 0; back = .FALSE.; wvert = .FALSE.; iconvect = 0
  CALL METINI(0,0,1.0,1,1,ngrd,ntim,plat,kyr,kmo,kda,khr,kmn,back,kvel,1,1,wvert,iconvect)
  nx = GRID(1,1)%NX; ny = GRID(1,1)%NY; lrec = FILE(1,1)%REC_LEN

  SELECT CASE (TRIM(mode))
  CASE ('grid')
    OPEN(20, FILE='grid_out.txt', STATUS='REPLACE')
    WRITE(20,'(A,1X,A4,1X,I6)') 'model_number', GRID(1,1)%MODEL_ID, GRID(1,1)%NUMBER
    WRITE(20,'(A,12(1X,ES16.8))') 'proj', GRID(1,1)%POLE_LAT, GRID(1,1)%POLE_LON, &
      GRID(1,1)%REF_LAT, GRID(1,1)%REF_LON, GRID(1,1)%SIZE, GRID(1,1)%ORIENT, &
      GRID(1,1)%TANG_LAT, GRID(1,1)%SYNC_XP, GRID(1,1)%SYNC_YP, GRID(1,1)%SYNC_LAT, &
      GRID(1,1)%SYNC_LON, GRID(1,1)%DUMMY
    WRITE(20,'(A,3(1X,I6))') 'nxyz', GRID(1,1)%NX, GRID(1,1)%NY, GRID(1,1)%NZ
    WRITE(20,'(A,4(1X,L1))') 'latlon_prime_gbldat_global', GRID(1,1)%LATLON, GRID(1,1)%PRIME, &
      GRID(1,1)%GBLDAT, GRID(1,1)%GLOBAL
    WRITE(20,'(A,9(1X,ES16.8))') 'gbase', GRID(1,1)%GBASE
    WRITE(20,'(A,6(1X,I8))') 'zflag_recper_delta_offset_reclen_endrec', DREC(1,1)%Z_FLAG, &
      DREC(1,1)%REC_PER, DREC(1,1)%DELTA, DREC(1,1)%OFFSET, FILE(1,1)%REC_LEN, FILE(1,1)%ENDREC
    WRITE(20,'(A,6(1X,I6))') 'first', FILE(1,1)%FIRST%YR, FILE(1,1)%FIRST%MO, FILE(1,1)%FIRST%DA, &
      FILE(1,1)%FIRST%HR, FILE(1,1)%FIRST%MN, FILE(1,1)%FIRST%IC
    WRITE(20,'(A,6(1X,I6))') 'last', FILE(1,1)%LAST%YR, FILE(1,1)%LAST%MO, FILE(1,1)%LAST%DA, &
      FILE(1,1)%LAST%HR, FILE(1,1)%LAST%MN, FILE(1,1)%LAST%IC
    WRITE(20,'(A,20(1X,L1))') 'flags_prss_shgt_zflg_qflg_uflg_tflg_mixd_tken_velv_dzdt', &
      DREC(1,1)%PRSS, DREC(1,1)%SHGT, DREC(1,1)%ZFLG, DREC(1,1)%QFLG, DREC(1,1)%UFLG, &
      DREC(1,1)%TFLG, DREC(1,1)%MIXD, DREC(1,1)%TKEN, DREC(1,1)%VELV, DREC(1,1)%DZDT
    WRITE(20,'(A,1X,I4)') 'wrf', DREC(1,1)%WRF
    DO l = 1, GRID(1,1)%NZ
      WRITE(20,'(A,1X,I3,1X,ES16.8,1X,I3,99(1X,A4,1X,I3))') 'level', l, DREC(1,1)%HEIGHT(l), &
        DREC(1,1)%NUM_VARB(l), (DREC(1,1)%VARB_ID(m,l), DREC(1,1)%CHK_SUM(m,l), m=1,DREC(1,1)%NUM_VARB(l))
    END DO
    READ(*,*) n
    DO k = 1, n
      READ(*,*) xp, yp
      IF (GRID(1,1)%LATLON) THEN
        CALL GBL2LL(1,1,xp,yp,clat,clon)
      ELSE
        CALL CXY2LL(GRID(1,1)%GBASE,xp,yp,clat,clon)
      END IF
      WRITE(20,'(A,4(1X,ES16.8))') 'xy2ll', xp, yp, clat, clon
    END DO
    READ(*,*) n
    DO k = 1, n
      READ(*,*) clat, clon
      IF (GRID(1,1)%LATLON) THEN
        CALL GBL2XY(1,1,clat,clon,xp,yp)
      ELSE
        CALL CLL2XY(GRID(1,1)%GBASE,clat,clon,xp,yp)
      END IF
      WRITE(20,'(A,4(1X,ES16.8))') 'll2xy', clat, clon, xp, yp
    END DO
    CLOSE(20)

  CASE ('labels')
    CALL do_labels()

  CASE ('values')
    CALL do_values()

  CASE ('column')
    CALL do_column()
  END SELECT

CONTAINS

  SUBROUTINE do_labels()
    CHARACTER(50) :: label
    CHARACTER(4)  :: varb, kvar, model
    CHARACTER(2)  :: cgrid
    CHARACTER(1), ALLOCATABLE :: cdata(:)
    CHARACTER(:), ALLOCATABLE :: header
    INTEGER :: irec, ios, iy, im, id, ih, ic, il, nexp, lenh, nndx, nhl1, nhl2, j, kol, nvar, nlvl
    INTEGER :: icx, mn, knx, kny, gnx, gny, gnz, zflag, ldat, chk
    REAL    :: prec, var1, f12(12), hgt
    ALLOCATE(cdata(nx*ny))
    ldat = nx*ny
    OPEN(10, FILE=TRIM(FILE(1,1)%DIR)//TRIM(FILE(1,1)%METEO), ACCESS='DIRECT', &
         FORM='UNFORMATTED', RECL=lrec, STATUS='OLD', ACTION='READ')
    OPEN(20, FILE='labels_out.txt', STATUS='REPLACE')
    irec = 0
    DO
      irec = irec + 1
      READ(10, REC=irec, IOSTAT=ios) label, cdata
      IF (ios /= 0) EXIT
      READ(label,'(5I2,2X,A2,A4)', IOSTAT=ios) iy, im, id, ih, ic, cgrid, kvar
      IF (kvar == 'INDX') THEN
        READ(label,'(5I2)') iy, im, id, ih, ic
        ALLOCATE(CHARACTER(108) :: header)
        DO j = 1, 108
          header(j:j) = cdata(j)
        END DO
        READ(header(1:108),'(A4,I3,I2,12F7.0,3I3,I2,I4)') model, icx, mn, f12, gnx, gny, gnz, zflag, lenh
        DEALLOCATE(header)
        ALLOCATE(CHARACTER(lenh) :: header)
        nndx = lenh/ldat + 1
        nhl1 = 1
        DO j = 1, nndx
          nhl2 = nhl1 + ldat - 1
          IF (j == nndx) nhl2 = nhl1 + (lenh - (nndx-1)*ldat) - 1
          READ(10, REC=irec+j-1) label, cdata
          DO k = nhl1, nhl2
            header(k:k) = cdata(k-nhl1+1)
          END DO
          nhl1 = nhl2 + 1
        END DO
        WRITE(20,'(A,1X,I8,6(1X,I4),1X,A2,1X,A4,1X,I4,1X,I3)') 'INDX', irec, iy, im, id, ih, ic, mn, cgrid, model, icx, nndx
        WRITE(20,'(A,12(1X,ES16.8))') 'proj', f12
        WRITE(20,'(A,5(1X,I6))') 'nxyz_flag_lenh', gnx, gny, gnz, zflag, lenh
        kol = 109
        nlvl = gnz
        DO l = 1, nlvl
          READ(header(kol:kol+7),'(F6.2,I2)') hgt, nvar
          WRITE(20,'(A,1X,I3,1X,ES16.8,1X,I3)', ADVANCE='NO') 'level', l, hgt, nvar
          kol = kol + 8
          DO m = 1, nvar
            READ(header(kol:kol+7),'(A4,I3)') varb, chk
            WRITE(20,'(1X,A4,1X,I3)', ADVANCE='NO') varb, chk
            kol = kol + 8
          END DO
          WRITE(20,*)
        END DO
        DEALLOCATE(header)
        irec = irec + nndx - 1
      ELSE
        READ(label,'(6I2,2X,A4,I4,2E14.7)', IOSTAT=ios) iy, im, id, ih, ic, il, varb, nexp, prec, var1
        IF (ios /= 0) THEN
          WRITE(20,'(A,1X,I8,1X,A)') 'BADLABEL', irec, label
        ELSE
          WRITE(20,'(A,1X,I8,6(1X,I4),1X,A4,1X,I4,2(1X,ES16.8))') 'REC', irec, iy, im, id, ih, ic, il, varb, nexp, prec, var1
        END IF
      END IF
    END DO
    CLOSE(10); CLOSE(20)
  END SUBROUTINE do_labels

  SUBROUTINE do_values()
    CHARACTER(50) :: label
    CHARACTER(4)  :: varb
    CHARACTER(1), ALLOCATABLE :: cdata(:)
    REAL, ALLOCATABLE :: rvar(:,:)
    INTEGER :: nrec, irec, iy, im, id, ih, ic, il, nexp, ksum, ios
    REAL    :: prec, var1
    ALLOCATE(cdata(nx*ny), rvar(nx,ny))
    OPEN(10, FILE=TRIM(FILE(1,1)%DIR)//TRIM(FILE(1,1)%METEO), ACCESS='DIRECT', &
         FORM='UNFORMATTED', RECL=lrec, STATUS='OLD', ACTION='READ')
    OPEN(20, FILE='values_out.bin', ACCESS='STREAM', FORM='UNFORMATTED', STATUS='REPLACE')
    OPEN(30, FILE='values_hdr.txt', STATUS='REPLACE')
    READ(*,*) nrec
    DO k = 1, nrec
      READ(*,*) irec
      READ(10, REC=irec) label, cdata
      READ(label,'(6I2,2X,A4,I4,2E14.7)', IOSTAT=ios) iy, im, id, ih, ic, il, varb, nexp, prec, var1
      ksum = -1
      CALL PAKINP(rvar, cdata, nx, ny, 1, 1, nx, ny, prec, nexp, var1, ksum)
      WRITE(20) rvar
      WRITE(30,'(I8,1X,A4,1X,I3,1X,I4,1X,I8)') irec, varb, il, ic, ksum
    END DO
    CLOSE(10); CLOSE(20); CLOSE(30)
  END SUBROUTINE do_values

  SUBROUTINE do_column()
    INCLUDE 'DEFARG4.INC'
    INTEGER, PARAMETER :: NL = 4000
    REAL,    PARAMETER :: DZ = 5.0, ZMDL = 25000.0
    CHARACTER(50) :: label
    CHARACTER(4)  :: varb
    CHARACTER(1), ALLOCATABLE :: cdata(:)
    REAL, ALLOCATABLE :: rvar(:,:), col(:,:,:), sfc(:,:)
    INTEGER, ALLOCATABLE :: ii(:), jj(:)
    INTEGER :: t, npt, irec, base, iy, im, id, ih, ic, il, nexp, ksum, nz, nzs, ip, kv, ktop, kk
    REAL    :: prec, var1
    ! upper fields: 1 TEMP 2 SPHU 3 RELH 4 PRES 5 HGTS 6 UWND 7 VWND 8 WWND
    ! surface: 1 PRSS 2 T02M 3 SHGT 4 U10M 5 V10M 6 TMPS
    CHARACTER(4), PARAMETER :: uname(8) = (/'TEMP','SPHU','RELH','PRES','HGTS','UWND','VWND','WWND'/)
    CHARACTER(4), PARAMETER :: sname(6) = (/'PRSS','T02M','SHGT','U10M','V10M','TMPS'/)
    LOGICAL :: have_u(8), have_s(6), qflg, tflg, uflg, pflg, sflg, zflg
    REAL, ALLOCATABLE :: psg(:), esg(:), p1(:), t1(:), q1(:), u1(:), v1(:), w1(:), e1(:), h1(:), x1(:), kz1(:), kh1(:), c1(:)
    REAL :: zsg(NL), pp(NL), uu(NL), vv(NL), ww(NL), tt(NL), zz(NL), rh(NL), den(NL), aa(NL)
    REAL :: ee(NL), hh(NL), xx(NL), kz2(NL), kh2(NL), c2(NL,9)
    REAL :: zsfc, p0, u0, v0, t0, z0, offset
    CHARACTER(64) :: fname

    READ(*,*) t
    READ(*,*) npt
    ALLOCATE(ii(npt), jj(npt))
    DO k = 1, npt
      READ(*,*) ii(k), jj(k)
    END DO
    nz = GRID(1,1)%NZ
    nzs = nz - 1
    ALLOCATE(cdata(nx*ny), rvar(1,1), col(npt,nz,8), sfc(npt,6))
    col = 0.0; sfc = 0.0; have_u = .FALSE.; have_s = .FALSE.
    OPEN(10, FILE=TRIM(FILE(1,1)%DIR)//TRIM(FILE(1,1)%METEO), ACCESS='DIRECT', &
         FORM='UNFORMATTED', RECL=lrec, STATUS='OLD', ACTION='READ')
    base = (t-1)*DREC(1,1)%REC_PER + 1   ! the time step's INDX record
    irec = base + (DREC(1,1)%REC_PER - SUM(DREC(1,1)%NUM_VARB(1:nz)))  ! first data record
    DO l = 1, nz
      DO m = 1, DREC(1,1)%NUM_VARB(l)
        READ(10, REC=irec) label, cdata
        READ(label,'(6I2,2X,A4,I4,2E14.7)') iy, im, id, ih, ic, il, varb, nexp, prec, var1
        IF (varb /= DREC(1,1)%VARB_ID(m,l) .OR. il+1 /= l) THEN
          WRITE(*,*) 'LAYOUT MISMATCH at record', irec, varb, il, DREC(1,1)%VARB_ID(m,l), l
          STOP 1
        END IF
        DO ip = 1, npt
          ksum = -1
          CALL PAKINP(rvar, cdata, nx, ny, ii(ip), jj(ip), 1, 1, prec, nexp, var1, ksum)
          IF (l == 1) THEN
            DO kv = 1, 6
              IF (varb == sname(kv)) THEN
                sfc(ip,kv) = rvar(1,1); have_s(kv) = .TRUE.
              END IF
            END DO
          ELSE
            DO kv = 1, 8
              IF (varb == uname(kv)) THEN
                col(ip,l,kv) = rvar(1,1); have_u(kv) = .TRUE.
              END IF
            END DO
            IF (varb == 'DIFW') col(ip,l,8) = col(ip,l,8) + rvar(1,1)   ! metinp: W = W + DIFW
          END IF
        END DO
        irec = irec + 1
      END DO
    END DO
    CLOSE(10)

    ALLOCATE(psg(nzs), esg(nzs), p1(nzs), t1(nzs), q1(nzs), u1(nzs), v1(nzs), w1(nzs), &
             e1(nzs), h1(nzs), x1(nzs), kz1(nzs), kh1(nzs), c1(nzs))
    DO k = 1, nzs
      psg(k) = DREC(1,1)%HEIGHT(k+1)
    END DO
    offset = GRID(1,1)%DUMMY
    qflg = have_u(2)
    tflg = have_s(2) .OR. have_s(6)
    uflg = have_s(4) .AND. have_s(5)
    pflg = have_s(1)
    sflg = have_s(3)
    zflg = have_u(4)
    DO ip = 1, npt
      WRITE(fname,'(A,I0,A)') 'raw_', ip, '.txt'
      OPEN(21, FILE=TRIM(fname), STATUS='REPLACE')
      WRITE(21,'(A,6(1X,ES16.8))') 'sfc', sfc(ip,:)
      DO l = 2, nz
        WRITE(21,'(I3,9(1X,ES16.8))') l, DREC(1,1)%HEIGHT(l), col(ip,l,:)
      END DO
      CLOSE(21)
      p1 = col(ip,2:nz,4); t1 = col(ip,2:nz,1); u1 = col(ip,2:nz,6); v1 = col(ip,2:nz,7)
      w1 = col(ip,2:nz,8)
      IF (qflg) THEN
        q1 = col(ip,2:nz,2)
      ELSE
        q1 = col(ip,2:nz,3)
      END IF
      e1 = 0.0; h1 = 0.0; x1 = 0.0; kz1 = 0.0; kh1 = 0.0; c1 = 0.0
      p0 = sfc(ip,1); zsfc = sfc(ip,3); u0 = sfc(ip,4); v0 = sfc(ip,5); z0 = 0.1
      IF (have_s(2)) THEN
        t0 = sfc(ip,2)
      ELSE
        t0 = sfc(ip,6)
      END IF
      DO k = 1, NL
        zsg(k) = 1.0 - (k*DZ)/(ZMDL - zsfc)
      END DO
      ktop = 0
      SELECT CASE (DREC(1,1)%Z_FLAG)
      CASE (1)
        CALL PRFSIG(.FALSE.,.FALSE.,.FALSE.,.FALSE.,.FALSE.,.FALSE.,.FALSE., &
             zflg,qflg,uflg,tflg,pflg,sflg,.FALSE.,GRID(1,1)%MODEL_ID == 'AWRF', &
             offset,ktop,zsfc,p0,u0,v0,t0,z0,nzs,psg,p1,u1,v1,w1,t1,q1, &
             e1,h1,x1,kz1,kh1,NL, &
             ZMDL,zsg,pp,uu,vv,ww,tt,zz,rh,den,aa,ee,hh,xx,kz2,kh2, &
             .FALSE.,.FALSE., &
             c1,c1,c1,c1,c1,c1,c1,c1,c1, &
             c2(:,1),c2(:,2),c2(:,3),c2(:,4),c2(:,5),c2(:,6),c2(:,7),c2(:,8),c2(:,9))
      CASE (2)
        p1 = col(ip,2:nz,5)   ! PRFPRS takes the heights (HGTS) in this slot
        CALL PRFPRS(.FALSE.,.FALSE.,.FALSE.,qflg,uflg,tflg,pflg,.FALSE.,sflg,ktop,zsfc, &
             p0,u0,v0,t0,z0,nzs,psg,p1,u1,v1,w1,t1,q1,e1,h1,x1,NL,ZMDL,zsg, &
             pp,uu,vv,ww,tt,zz,rh,den,aa,ee,hh,xx)
      CASE (4)
        CALL PRFECM(.FALSE.,.FALSE.,.FALSE.,qflg,uflg,tflg,pflg,.FALSE.,sflg,ktop,zsfc, &
             p0,u0,v0,t0,z0,nzs,psg,esg,u1,v1,w1,t1,q1,e1,h1,x1,NL,ZMDL,zsg, &
             pp,uu,vv,ww,tt,zz,rh,den,aa,ee,hh,xx)
      CASE DEFAULT
        WRITE(*,*) 'flag not handled', DREC(1,1)%Z_FLAG
        STOP 2
      END SELECT
      WRITE(fname,'(A,I0,A)') 'column_', ip, '.txt'
      OPEN(21, FILE=TRIM(fname), STATUS='REPLACE')
      WRITE(21,'(A,1X,I3,1X,I5,1X,I5,6(1X,L1),2(1X,ES16.8),1X,I6)') '#', DREC(1,1)%Z_FLAG, ii(ip), jj(ip), &
        zflg, qflg, uflg, tflg, pflg, sflg, zsfc, p0, ktop
      DO kk = 1, NL
        WRITE(21,'(F10.2,5(1X,ES16.8))') zz(kk), pp(kk), aa(kk), uu(kk), vv(kk), den(kk)
      END DO
      CLOSE(21)
    END DO
  END SUBROUTINE do_column

END PROGRAM hyref
