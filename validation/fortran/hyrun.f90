! HYSPLIT's values at points, on HYSPLIT's own library, on HYSPLIT's internal vertical grid (#144).
!
! stdin: directory (ending in /); file name; t (time step, 1-based, file order);
!        nh; nh heights (m AGL at the point, increasing); zdata (CONTROL's top of model domain);
!        npt; npt lines "x y" (1-based mass-grid positions, fractional).
! As hymodelc does: ZMDL = MAX(|zdata|, 25000) (RUNSET); METLVL(1, zdata) picks the internal grid
! (AA, BB, CC, NLVL; COMMON /ZZTOKK/); ZSG(N) = 1 - (AA N^2 + BB N + CC) / ZMDL. At each of the four
! nodes the flag's profile routine fills the NLVL internal levels (heights (1 - ZSG) (ZMDL - zsfc_node),
! extrapolating above the data, PRFEND / prfsig's tail). A particle at height z above its terrain zpt
! (ADV3NT of SHGT) has sigma zz = 1 - z / (ZMDL - zpt); ZINDEX gives its vertical index ZX and ADV3NT
! blends the 2x2xNLVL profiles (the advection path). Also the PARTICLE_STILT.DAT path of hymodelc
! (KBB/KTT/ZF linear in sigma between internal levels, level 1 below it; ZPOS clamped to ZSG(NLVL)).
! For time step t: each record unpacked with PAKINP (full grid, once), its values kept at the
! nodes the points need; metinp.f's conversions (WRF = 3: UWND*MSFU/MUU0, VWND*MSFV/MUV0;
! THET*(PRES/1000)**0.286; DIFW added); then at each of the four nodes around a point the flag's
! profile routine (PRFSIG flag 1, PRFPRS flag 2 with HGTS, PRFECM flag 4) with output levels
! at the requested heights, and ADV3NT on the 2x2 node profiles: T and P at (x, y), U at
! (x+SUX, y+SUY), V at (x+SVX, y+SVY) from METINI's /STAGGER/.
! Output: out.txt, per point and height: ip z U V T P zsfc_point zx Ulin Vlin Tlin Plin;
!   levels.txt: nlvl aa bb cc zmdl, then each internal level's height (1 - ZSG) ZMDL;
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
  INCLUDE 'zindex.inc'
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

  REAL :: ZMDL, zdata, qa, qb, qc, zx, zpos, zf, sfcl
  COMMON /ZZTOKK/ qa, qb, qc
  INTEGER :: nlvl, ksfc, kbb, ktt, kzz
  REAL :: ul, vl, tl, pl
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
  READ(*,*) zdata
  ZMDL = MAX(ABS(zdata), 25000.0)
  CALL METLVL(1, zdata, nlvl, ksfc, sfcl)
  mode = 1
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
  ALLOCATE(zsg(nlvl), pp(nlvl), uu(nlvl), vv(nlvl), ww(nlvl), tt(nlvl), zz(nlvl), rh(nlvl), den(nlvl), &
           aa(nlvl), ee(nlvl), hh(nlvl), xx(nlvl), kz2(nlvl), kh2(nlvl), c2(nlvl,9))
  ALLOCATE(sU(2,2,nlvl), sV(2,2,nlvl), sT(2,2,nlvl), sP(2,2,nlvl), sZ(2,2,1))
  DO k = 1, nlvl
    zsg(k) = 1.0 - (qa*FLOAT(k*k) + qb*FLOAT(k) + qc)/ZMDL
  END DO
  OPEN(22, FILE='levels.txt', STATUS='REPLACE')
  WRITE(22,'(I6,4(1X,ES16.8))') nlvl, qa, qb, qc, ZMDL
  DO k = 1, nlvl
    WRITE(22,'(ES16.8)') (1.0 - zsg(k))*ZMDL
  END DO
  CLOSE(22)
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
  OPEN(23, FILE='ktop.txt', STATUS='REPLACE')
  WRITE(20,'(A,4(1X,F6.3),3(1X,I3))') '# SUX SUY SVX SVY WRF FLAG MODE', SUX, SUY, SVX, SVY, &
       DREC(1,1)%WRF, flag, mode
  DO ip = 1, npt
    ! the point's terrain, as the model's ZSFC at the particle (ADV3NT of SHGT on the mass nodes)
    n = (ip-1)*12
    ! the terrain array at the mass nodes: SHGT, else PRFA's estimate from PRSS, which PRFSIG and
    ! PRFECM write back into ZT(I,J) (prfcom.f), so the particle's ZTER is interpolated from it
    DO k = 1, 4
      zs2 = sfc(n+k,3)
      IF (.NOT. sflg .AND. flag /= 2) THEN
        p02 = sfc(n+k,1)
        CALL PRFA(pflg, sflg, up(n+k,1,1), zs2, p02, pb)
      END IF
      sZ(1 + MOD(k-1, 2), 1 + (k-1)/2, 1) = zs2
    END DO
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
        ktop = 0
        SELECT CASE (flag)
        CASE (1)
          CALL PRFSIG(.FALSE.,.FALSE.,.FALSE.,.FALSE.,.FALSE.,.FALSE.,.FALSE., &
               zflg,qflg,uflg,tflg,pflg,sflg,.FALSE.,GRID(1,1)%MODEL_ID == 'AWRF', &
               offset,ktop,zsfc,p0,u0,v0,t0,z0,nzs,psg,p1,u1,v1,w1,t1,q1, &
               e1,h1,x1,kz1,kh1,nlvl, &
               ZMDL,zsg,pp,uu,vv,ww,tt,zz,rh,den,aa,ee,hh,xx,kz2,kh2, &
               .FALSE.,.FALSE., &
               c1,c1,c1,c1,c1,c1,c1,c1,c1, &
               c2(:,1),c2(:,2),c2(:,3),c2(:,4),c2(:,5),c2(:,6),c2(:,7),c2(:,8),c2(:,9))
        CASE (2)
          p1 = up(n,:,10)   ! PRFPRS takes the heights (HGTS) in this slot
          CALL PRFPRS(.FALSE.,.FALSE.,.FALSE.,qflg,uflg,tflg,pflg,.FALSE.,sflg,ktop,zsfc, &
               p0,u0,v0,t0,z0,nzs,psg,p1,u1,v1,w1,t1,q1,e1,h1,x1,nlvl,ZMDL,zsg, &
               pp,uu,vv,ww,tt,zz,rh,den,aa,ee,hh,xx)
        CASE (4)
          CALL PRFECM(.FALSE.,.FALSE.,.FALSE.,qflg,uflg,tflg,pflg,.FALSE.,sflg,ktop,zsfc, &
               p0,u0,v0,t0,z0,nzs,psg,esg,u1,v1,w1,t1,q1,e1,h1,x1,nlvl,ZMDL,zsg, &
               pp,uu,vv,ww,tt,zz,rh,den,aa,ee,hh,xx)
        CASE DEFAULT
          WRITE(*,*) 'flag not handled', flag
          STOP 2
        END SELECT
        WRITE(23,'(4I6,1X,F9.2)') ip, ipos, k, ktop, zsfc
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
      zpos = 1.0 - hz(k)/(ZMDL - zpt)
      CALL ZINDEX(nlvl, ZMDL, zpos, zsg, zx)
      ! hymodelc's PARTICLE_STILT.DAT path
      zpos = MIN(1.0, MAX(zpos, zsg(nlvl)))
      kzz = 1
      DO WHILE (zsg(kzz+1) .GT. zpos)
        kzz = kzz + 1
      END DO
      ktt = kzz + 1; kbb = ktt - 1
      zf = (zsg(kbb) - zpos)/(zsg(kbb) - zsg(ktt))
      IF (zpos > zsg(1)) zf = 0.0
      i1 = MAX(1, MIN(INT(xs(1)), nx)); j1 = MAX(1, MIN(INT(ys(1)), ny))
      xl = xs(1)-i1+1; yl = ys(1)-j1+1
      CALL ADV3NT(sT, xl, yl, zx, c, .FALSE., 2, 2)
      CALL ADV3NT(sP, xl, yl, zx, d, .FALSE., 2, 2)
      CALL ADV3NT(sT, xl, yl, REAL(kbb), tl, .FALSE., 2, 2)
      CALL ADV3NT(sT, xl, yl, REAL(ktt), pb, .FALSE., 2, 2)
      tl = tl + zf*(pb - tl)
      CALL ADV3NT(sP, xl, yl, REAL(kbb), pl, .FALSE., 2, 2)
      CALL ADV3NT(sP, xl, yl, REAL(ktt), pb, .FALSE., 2, 2)
      pl = pl + zf*(pb - pl)
      i1 = MAX(1, MIN(INT(xs(2)), nx)); j1 = MAX(1, MIN(INT(ys(2)), ny))
      CALL ADV3NT(sU, xs(2)-i1+1, ys(2)-j1+1, zx, a, .FALSE., 2, 2)
      CALL ADV3NT(sU, xs(2)-i1+1, ys(2)-j1+1, REAL(kbb), ul, .FALSE., 2, 2)
      CALL ADV3NT(sU, xs(2)-i1+1, ys(2)-j1+1, REAL(ktt), pb, .FALSE., 2, 2)
      ul = ul + zf*(pb - ul)
      i1 = MAX(1, MIN(INT(xs(3)), nx)); j1 = MAX(1, MIN(INT(ys(3)), ny))
      CALL ADV3NT(sV, xs(3)-i1+1, ys(3)-j1+1, zx, b, .FALSE., 2, 2)
      CALL ADV3NT(sV, xs(3)-i1+1, ys(3)-j1+1, REAL(kbb), vl, .FALSE., 2, 2)
      CALL ADV3NT(sV, xs(3)-i1+1, ys(3)-j1+1, REAL(ktt), pb, .FALSE., 2, 2)
      vl = vl + zf*(pb - vl)
      WRITE(20,'(I6,1X,F9.2,4(1X,ES16.8),1X,F9.2,1X,F10.5,4(1X,ES16.8))') ip, hz(k), a, b, c, d, zpt, &
           zx, ul, vl, tl, pl
    END DO
  END DO
  CLOSE(20); CLOSE(21); CLOSE(23)
END PROGRAM hyorder
