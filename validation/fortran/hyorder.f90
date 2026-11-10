! HYSPLIT's values at points between grid nodes, in HYSPLIT's order, on HYSPLIT's own library.
!
! stdin: directory (ending in /); file name; t (time step, 1-based, file order);
!        nh; nh heights (m AGL, increasing); mode (0: same AGL at every node;
!        1: HYSPLIT's terrain-following sigma, the point's height scaled to each node by
!        (ZMDL - zsfc_node) / (ZMDL - zsfc_point), zsfc_point = ADV3NT of SHGT); npt; npt lines
!        "x y" (1-based mass-grid positions, fractional).
! For time step t: each record unpacked with PAKINP (full grid, once), its values kept at the
! nodes the points need; metinp.f's conversions (WRF = 3: UWND*MSFU/MUU0, VWND*MSFV/MUV0;
! THET*(PRES/1000)**0.286; DIFW added); then at each of the four nodes around a point the flag's
! profile routine (PRFSIG flag 1, PRFPRS flag 2 with HGTS, PRFECM flag 4) with output levels
! at the requested heights, and ADV3NT on the 2x2 node profiles: T and P at (x, y), U at
! (x+SUX, y+SUY), V at (x+SVX, y+SVY) from METINI's /STAGGER/.
! Output: out.txt, per point and height: ip z U V T P zsfc_point;
!   points.txt, per point: ip x y lat lon zsfc_point (GBL2LL or CXY2LL)
PROGRAM hyorder
  USE prfmod, ONLY: PRFA
  IMPLICIT NONE
  INCLUDE 'limits.inc'
  INCLUDE 'DEFGRID.INC'
  INCLUDE 'gblgrd.inc'
  INCLUDE 'DEFARG1.INC'
  INCLUDE 'DEFARG4.INC'
  INCLUDE 'tminit.inc'
  INCLUDE 'adv3nt.inc'
  INCLUDE 'gbl2ll.inc'
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

  REAL       :: SUX,SUY,SVX,SVY
  COMMON /STAGGER/ SUX,SUY,SVX,SVY

  REAL, PARAMETER :: ZMDL = 25000.0
  ! surface: 1 PRSS 2 T02M 3 SHGT 4 U10M 5 V10M 6 TMPS 7 MSFU 8 MSFV 9 MSFT 10 MUU0 11 MUV0
  CHARACTER(4), PARAMETER :: sname(11) = (/'PRSS','T02M','SHGT','U10M','V10M','TMPS', &
                                           'MSFU','MSFV','MSFT','MUU0','MUV0'/)
  ! upper: 1 TEMP 2 THET 3 SPHU 4 RELH 5 PRES 6 UWND 7 VWND 8 WWND 9 DIFW 10 HGTS
  CHARACTER(4), PARAMETER :: uname(10) = (/'TEMP','THET','SPHU','RELH','PRES','UWND','VWND', &
                                           'WWND','DIFW','HGTS'/)
  CHARACTER(50) :: label
  CHARACTER(4)  :: varb
  CHARACTER(1), ALLOCATABLE :: cdata(:)
  REAL, ALLOCATABLE :: rvar(:,:), px(:), py(:), hz(:)
  INTEGER, ALLOCATABLE :: ni(:), nj(:)          ! node list: 12 per point (mass, U, V; 4 each)
  REAL, ALLOCATABLE :: sfc(:,:), up(:,:,:)      ! (node, var), (node, level, var)
  REAL, ALLOCATABLE :: psg(:), esg(:), p1(:), t1(:), q1(:), u1(:), v1(:), w1(:), e1(:), h1(:), &
                       x1(:), kz1(:), kh1(:), c1(:)
  REAL, ALLOCATABLE :: zsg(:), pp(:), uu(:), vv(:), ww(:), tt(:), zz(:), rh(:), den(:), aa(:), &
                       ee(:), hh(:), xx(:), kz2(:), kh2(:), c2(:,:)
  REAL, ALLOCATABLE :: sU(:,:,:), sV(:,:,:), sT(:,:,:), sP(:,:,:), sZ(:,:,:)
  REAL :: zsfc, p0, u0, v0, t0, z0, offset, plat, prec, var1, a, b, c, d, zpt, xs(3), ys(3)
  REAL :: xl, yl, clat, clon, zs2, p02, pb
  LOGICAL :: have_s(11), have_u(10), back, wvert, qflg, tflg, uflg, pflg, sflg, zflg, thetflg
  INTEGER :: iconvect, ngrd, ntim, kvel, kyr, kmo, kda, khr, kmn, nx, ny, nz, nzs, lrec, nh, mode
  INTEGER :: t, npt, nn, irec, base, iy, im, id, ih, ic, il, nexp, ksum, l, m, k, kv, i, j, ip, ktop
  INTEGER :: ipos, ii, jj, n, i1, j1, flag

  READ(*,'(A)') FILE(1,1)%DIR
  FILE(1,1)%DIR = ADJUSTL(FILE(1,1)%DIR)
  READ(*,'(A)') FILE(1,1)%METEO
  OPEN(70, FILE='MESSAGE')
  OPEN(85, FILE='WARNING')
  CALL TMINIT
  ngrd = 1; ntim = 1; kvel = 0; back = .FALSE.; wvert = .FALSE.; iconvect = 0
  CALL METINI(0,0,1.0,1,1,ngrd,ntim,plat,kyr,kmo,kda,khr,kmn,back,kvel,1,1,wvert,iconvect)
  nx = GRID(1,1)%NX; ny = GRID(1,1)%NY; lrec = FILE(1,1)%REC_LEN
  nz = GRID(1,1)%NZ; nzs = nz - 1
  flag = DREC(1,1)%Z_FLAG

  READ(*,*) t
  READ(*,*) nh
  ALLOCATE(hz(nh))
  READ(*,*) hz
  READ(*,*) mode
  READ(*,*) npt
  ALLOCATE(px(npt), py(npt))
  DO k = 1, npt
    READ(*,*) px(k), py(k)
  END DO

  ! the nodes: for each point and each of its three positions (mass, U, V), the four around it
  nn = 12*npt
  ALLOCATE(ni(nn), nj(nn))
  DO ip = 1, npt
    xs = (/ px(ip), px(ip)+SUX, px(ip)+SVX /)
    ys = (/ py(ip), py(ip)+SUY, py(ip)+SVY /)
    DO ipos = 1, 3
      i1 = MAX(1, MIN(INT(xs(ipos)), nx)); j1 = MAX(1, MIN(INT(ys(ipos)), ny))
      n = (ip-1)*12 + (ipos-1)*4
      ni(n+1) = i1;               nj(n+1) = j1
      ni(n+2) = MIN(i1+1, nx);    nj(n+2) = j1
      ni(n+3) = i1;               nj(n+3) = MIN(j1+1, ny)
      ni(n+4) = MIN(i1+1, nx);    nj(n+4) = MIN(j1+1, ny)
    END DO
  END DO

  ALLOCATE(cdata(nx*ny), rvar(nx,ny), sfc(nn,11), up(nn,nzs,10))
  sfc = 0.0; up = 0.0; have_s = .FALSE.; have_u = .FALSE.
  OPEN(10, FILE=TRIM(FILE(1,1)%DIR)//TRIM(FILE(1,1)%METEO), ACCESS='DIRECT', &
       FORM='UNFORMATTED', RECL=lrec, STATUS='OLD', ACTION='READ')
  base = (t-1)*DREC(1,1)%REC_PER + 1
  irec = base + (DREC(1,1)%REC_PER - SUM(DREC(1,1)%NUM_VARB(1:nz)))
  DO l = 1, nz
    DO m = 1, DREC(1,1)%NUM_VARB(l)
      READ(10, REC=irec) label, cdata
      READ(label,'(6I2,2X,A4,I4,2E14.7)') iy, im, id, ih, ic, il, varb, nexp, prec, var1
      IF (varb /= DREC(1,1)%VARB_ID(m,l) .OR. il+1 /= l) THEN
        WRITE(*,*) 'LAYOUT MISMATCH at record', irec, varb, il, DREC(1,1)%VARB_ID(m,l), l
        STOP 1
      END IF
      irec = irec + 1
      IF (l == 1) THEN
        kv = FINDLOC(sname, varb, 1)
        IF (kv == 0) CYCLE
        ksum = -1
        CALL PAKINP(rvar, cdata, nx, ny, 1, 1, nx, ny, prec, nexp, var1, ksum)
        DO n = 1, nn
          sfc(n,kv) = rvar(ni(n), nj(n))
        END DO
        have_s(kv) = .TRUE.
      ELSE
        kv = FINDLOC(uname, varb, 1)
        IF (kv == 0) CYCLE
        ksum = -1
        CALL PAKINP(rvar, cdata, nx, ny, 1, 1, nx, ny, prec, nexp, var1, ksum)
        DO n = 1, nn
          up(n,l-1,kv) = rvar(ni(n), nj(n))
        END DO
        have_u(kv) = .TRUE.
      END IF
    END DO
  END DO
  CLOSE(10)

  ! metinp.f's conversions, at each node
  thetflg = have_u(2) .AND. .NOT. have_u(1)
  DO l = 1, nzs
    IF (DREC(1,1)%WRF.EQ.3) THEN
      up(:,l,6) = up(:,l,6)*sfc(:,7)/sfc(:,10)
      up(:,l,7) = up(:,l,7)*sfc(:,8)/sfc(:,11)
    END IF
    IF (have_u(9)) up(:,l,8) = up(:,l,8) + up(:,l,9)
    IF (thetflg) up(:,l,1) = up(:,l,2)*(up(:,l,5)/1000.0)**0.286
  END DO

  ALLOCATE(psg(nzs), esg(nzs), p1(nzs), t1(nzs), q1(nzs), u1(nzs), v1(nzs), w1(nzs), &
           e1(nzs), h1(nzs), x1(nzs), kz1(nzs), kh1(nzs), c1(nzs))
  ALLOCATE(zsg(nh), pp(nh), uu(nh), vv(nh), ww(nh), tt(nh), zz(nh), rh(nh), den(nh), aa(nh), &
           ee(nh), hh(nh), xx(nh), kz2(nh), kh2(nh), c2(nh,9))
  ALLOCATE(sU(2,2,nh), sV(2,2,nh), sT(2,2,nh), sP(2,2,nh), sZ(2,2,1))
  DO k = 1, nzs
    psg(k) = DREC(1,1)%HEIGHT(k+1)
  END DO
  esg = 0.0
  offset = GRID(1,1)%DUMMY
  qflg = have_u(3)
  tflg = have_s(2) .OR. have_s(6)
  uflg = have_s(4) .AND. have_s(5)
  pflg = have_s(1)
  sflg = have_s(3)
  zflg = have_u(5)

  OPEN(20, FILE='out.txt', STATUS='REPLACE')
  OPEN(21, FILE='points.txt', STATUS='REPLACE')
  WRITE(20,'(A,4(1X,F6.3),3(1X,I3))') '# SUX SUY SVX SVY WRF FLAG MODE', SUX, SUY, SVX, SVY, &
       DREC(1,1)%WRF, flag, mode
  DO ip = 1, npt
    ! the point's terrain, as the model's ZSFC at the particle (ADV3NT of SHGT on the mass nodes)
    n = (ip-1)*12
    sZ(1,1,1) = sfc(n+1,3); sZ(2,1,1) = sfc(n+2,3); sZ(1,2,1) = sfc(n+3,3); sZ(2,2,1) = sfc(n+4,3)
    i1 = MAX(1, MIN(INT(px(ip)), nx)); j1 = MAX(1, MIN(INT(py(ip)), ny))
    CALL ADV3NT(sZ, px(ip)-i1+1, py(ip)-j1+1, 1.0, zpt, .FALSE., 2, 2)
    IF (GRID(1,1)%LATLON) THEN
      CALL GBL2LL(1,1,px(ip),py(ip),clat,clon)
    ELSE
      CALL CXY2LL(GRID(1,1)%GBASE,px(ip),py(ip),clat,clon)
    END IF
    WRITE(21,'(I6,4(1X,ES18.10),1X,F9.2)') ip, px(ip), py(ip), clat, clon, zpt
    xs = (/ px(ip), px(ip)+SUX, px(ip)+SVX /)
    ys = (/ py(ip), py(ip)+SUY, py(ip)+SVY /)
    DO ipos = 1, 3
      DO k = 1, 4
        n = (ip-1)*12 + (ipos-1)*4 + k
        ii = 1 + MOD(k-1, 2); jj = 1 + (k-1)/2
        p1 = up(n,:,5); t1 = up(n,:,1); u1 = up(n,:,6); v1 = up(n,:,7); w1 = up(n,:,8)
        IF (qflg) THEN
          q1 = up(n,:,3)
        ELSE
          q1 = up(n,:,4)
        END IF
        e1 = 0.0; h1 = 0.0; x1 = 0.0; kz1 = 0.0; kh1 = 0.0; c1 = 0.0
        p0 = sfc(n,1); zsfc = sfc(n,3); u0 = sfc(n,4); v0 = sfc(n,5); z0 = 0.1
        IF (have_s(2)) THEN
          t0 = sfc(n,2)
        ELSE
          t0 = sfc(n,6)
        END IF
        IF (.NOT. sflg .AND. flag /= 2) THEN
          ! PRFSIG/PRFECM estimate the terrain from PRSS (PRFA) and place their output
          ! levels above it: set the levels from the same estimate
          zs2 = zsfc; p02 = p0
          CALL PRFA(pflg, sflg, t1(1), zs2, p02, pb)
        ELSE
          zs2 = zsfc
        END IF
        IF (mode == 0) THEN
          zsg = 1.0 - hz/(ZMDL - zs2)
        ELSE
          zsg = 1.0 - hz/(ZMDL - zpt)
        END IF
        ktop = 0
        SELECT CASE (flag)
        CASE (1)
          CALL PRFSIG(.FALSE.,.FALSE.,.FALSE.,.FALSE.,.FALSE.,.FALSE.,.FALSE., &
               zflg,qflg,uflg,tflg,pflg,sflg,.FALSE.,GRID(1,1)%MODEL_ID == 'AWRF', &
               offset,ktop,zsfc,p0,u0,v0,t0,z0,nzs,psg,p1,u1,v1,w1,t1,q1, &
               e1,h1,x1,kz1,kh1,nh, &
               ZMDL,zsg,pp,uu,vv,ww,tt,zz,rh,den,aa,ee,hh,xx,kz2,kh2, &
               .FALSE.,.FALSE., &
               c1,c1,c1,c1,c1,c1,c1,c1,c1, &
               c2(:,1),c2(:,2),c2(:,3),c2(:,4),c2(:,5),c2(:,6),c2(:,7),c2(:,8),c2(:,9))
        CASE (2)
          p1 = up(n,:,10)   ! PRFPRS takes the heights (HGTS) in this slot
          CALL PRFPRS(.FALSE.,.FALSE.,.FALSE.,qflg,uflg,tflg,pflg,.FALSE.,sflg,ktop,zsfc, &
               p0,u0,v0,t0,z0,nzs,psg,p1,u1,v1,w1,t1,q1,e1,h1,x1,nh,ZMDL,zsg, &
               pp,uu,vv,ww,tt,zz,rh,den,aa,ee,hh,xx)
        CASE (4)
          CALL PRFECM(.FALSE.,.FALSE.,.FALSE.,qflg,uflg,tflg,pflg,.FALSE.,sflg,ktop,zsfc, &
               p0,u0,v0,t0,z0,nzs,psg,esg,u1,v1,w1,t1,q1,e1,h1,x1,nh,ZMDL,zsg, &
               pp,uu,vv,ww,tt,zz,rh,den,aa,ee,hh,xx)
        CASE DEFAULT
          WRITE(*,*) 'flag not handled', flag
          STOP 2
        END SELECT
        IF (ipos == 1) THEN
          sT(ii,jj,:) = aa; sP(ii,jj,:) = pp
          IF (SUX == 0.0 .AND. SUY == 0.0) sU(ii,jj,:) = uu
          IF (SVX == 0.0 .AND. SVY == 0.0) sV(ii,jj,:) = vv
        ELSE IF (ipos == 2) THEN
          sU(ii,jj,:) = uu
        ELSE
          sV(ii,jj,:) = vv
        END IF
      END DO
    END DO
    DO k = 1, nh
      i1 = MAX(1, MIN(INT(xs(1)), nx)); j1 = MAX(1, MIN(INT(ys(1)), ny))
      xl = xs(1)-i1+1; yl = ys(1)-j1+1
      CALL ADV3NT(sT, xl, yl, REAL(k), c, .FALSE., 2, 2)
      CALL ADV3NT(sP, xl, yl, REAL(k), d, .FALSE., 2, 2)
      i1 = MAX(1, MIN(INT(xs(2)), nx)); j1 = MAX(1, MIN(INT(ys(2)), ny))
      CALL ADV3NT(sU, xs(2)-i1+1, ys(2)-j1+1, REAL(k), a, .FALSE., 2, 2)
      i1 = MAX(1, MIN(INT(xs(3)), nx)); j1 = MAX(1, MIN(INT(ys(3)), ny))
      CALL ADV3NT(sV, xs(3)-i1+1, ys(3)-j1+1, REAL(k), b, .FALSE., 2, 2)
      WRITE(20,'(I6,1X,F9.2,4(1X,ES16.8),1X,F9.2)') ip, hz(k), a, b, c, d, zpt
    END DO
  END DO
  CLOSE(20); CLOSE(21)
END PROGRAM hyorder
