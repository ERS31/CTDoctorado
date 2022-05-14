      program ejemplo
      implicit double precision(a-h,o-z)
      parameter(nmax=2000)
      integer iesp(nmax)
      real*8 rx(nmax),ry(nmax),rz(nmax)
      real*8 vx(nmax),vy(nmax),vz(nmax)
      real*8 fx(nmax),fy(nmax),fz(nmax)
      real*8 dx(nmax),dy(nmax),dz(nmax)
      real*8 ox(nmax),oy(nmax),oz(nmax)
      real*8 oxr(nmax),oyr(nmax),ozr(nmax)
      real*8 dip(nmax),amasa(nmax)
      real*8 onorma(nmax)
      real*8 dt
      real*8 ra
      real*8 normaf(nmax)
      character*1 cc


      open(86,file='datos18o.dat')
      rewind (86)
      open(88,file='datos.dat')
      rewind (88)
      open(90,file='datos18t.dat')
      rewind (90)
      cc=','

      call param(nmax,np,dip,amasa,dif0,campo,temp)
      write(6,*)'numero de particulas',np
      write(6,*)'coeficiente de autodif trasl dif0 ',dif0
      write(6,*)'intensidad de campo ',campo 
      write(6,*)'temperatura', temp

c Coeficiente de autodifusion rotacional

      difr0 = 3.0d0*dif0
      indif0= 1.0d0/dif0
      indifr0=1.0d0/difr0
 
      write(6,*)'coeficiente de autodif rotac difr0',difr0

      write(6,*) 'part    dipolo     masa'
      do i=1,np
        write(6,'(i6,2f12.6)')i,dip(i),amasa(i)
      enddo

      call configuracioninicial(nmax,n,boxx,boxy,boxz,rx,ry,rz,vx,vy,vz,
     & ox,oy,oz)

      write(6,*)'numero de particulas en 1particula0.dat',n
      if(n.ne.np) then 
        write(6,*) 'modificar np en paramTyR.dat'
        stop
      endif

      write(6,*)'lados de la celda',boxx,boxy,boxz
      do i=1,n
        write(6,'(i6,9f12.6)') i,rx(i),ry(i),rz(i),vx(i),vy(i),vz(i),
     & ox(i),oy(i),oz(i)
      enddo

      write(6,*) 'norma de la orientacion inicial'

      call norma(nmax,n,ox,oy,oz,onorma)

      do i=1,n
       write(6,*) i, onorma(i)
      enddo

      call paramdb(npasos,dt,nmovie)

      call fuerzas(nmax,n,campo,rx,ry,rz,fx,fy,fz)

c se realiza la dinamica por npasos

      do ipaso=1,npasos
        if(ipaso.eq.1) call gro(ipaso,nmax,n,rx,ry,rz,boxx,boxy,boxz,
     &  iesp)

        call posiciones(nmax,n,amasa,dt,rx,ry,rz,fx,fy,fz,dx,dy,dz,
     &  dif0)

        call orientacionaleatoria(nmax,n,dt,oxr,oyr,ozr,ox,oy,oz,
     & campo,difr0,normaf,np,amasa,temp)

ccc  call orientacionaleatoria()
c deben entrar variables aleatorias a la rutina 
c para que se calculen las orientaciones aleatorias que deben seguir
c una distribucion gaussiana con media cero y varianza 2(D_r)(delta t)

ccc        call torcas( )


ccc        call orientaciones( )


c erwin      write(6,'("paso----",i6)')ipaso
        do i=1,n
c erwin       write(6,'("posicion",i5,3f12.6)')i, rx(i),ry(i),rz(i)
        enddo

        if(mod(ipaso,nmovie).eq.0) then 
          call gro(ipaso,nmax,n,rx,ry,rz,boxx,boxy,boxz,iesp)
        endif

        call fuerzas(nmax,n,campo,rx,ry,rz,fx,fy,fz)

      enddo

      write(6,*)
      write(6,*)'--- fin de la dinamica ---'
      write(6,*)'numero de pasos',npasos
      write(6,*)'paso de integracion',dt
      write(6,*)'numero de particulas',n
      write(6,*)'lados de la celda',boxx,boxy,boxz

c      call promedios

      stop
      end

      subroutine param(nmax,np,dip,amasa,dif0,campo,temp)
      implicit double precision (a-h,o-z)
      real*8 dip(nmax),amasa(nmax)
      character*30 name

      write(6,*) 'datos de las particulas'
      open(11,file='paramTyR.dat')

      rewind 11
      read(11,*) np
      read(11,*) dipolo
      read(11,*) dif0
      read(11,*) xmasa
      read(11,*) campo
      read(11,*) temp

      do i=1,np
       dip(i)=dipolo
       amasa(i)=xmasa
      enddo

      close (11)
      return
      end

      subroutine configuracioninicial(nmax,n,boxx,boxy,boxz,rx,ry,rz,
     & vx,vy,vz,ox,oy,oz)
      implicit double precision (a-h,o-z)
      real*8 rx(nmax),ry(nmax),rz(nmax)
      real*8 vx(nmax),vy(nmax),vz(nmax)
      real*8 ox(nmax),oy(nmax),oz(nmax)
      character*30 name

      write(6,*) 'nombre del archivo de entrada: 1particula0.dat'
      read(5,'(A)') name
      open(12,file=name)

      rewind 12
      read(12,*) n
      read(12,*) boxx,boxy,boxz

      do i=1,n
         read(12,*)j,rx(i),ry(i),rz(i),vx(i),vy(i),vz(i),
     &   ox(i),oy(i),oz(i)
      enddo

      close (12)
      return
      end
      
      subroutine paramdb(npasos,dt,nmovie)
      implicit double precision (a-h,o-z)
      character*30 namedb
      
      write(6,*) 'archivo con parametros de DB: db.dat'
      read(5,'(A)') namedb
      open(13,file=namedb)

      read(13,*) npasos 
      write(6,*) 'numero de pasos     =',npasos 
      read(13,*) dt
      write(6,*) 'paso de integracion =',npasos 
      read(13,*) nmovie
      write(6,*) 'frecuencia para guardar posicion =',nmovie

      close (13)
      return
      end

      subroutine fuerzas(nmax,n,campo,rx,ry,rz,fx,fy,fz)
      implicit double precision (a-h,o-z)
      real*8 rx(nmax),ry(nmax),rz(nmax)
      real*8 fx(nmax),fy(nmax),fz(nmax)

      fact = campo

c Si una particula esta en el origen, la interaccion 
c campo-particula diverge

      do i=1,n
        r = dsqrt(rx(i)**2 + ry(i)**2 + rz(i)**2)
        rinv = 1.0d0/r
        r3 = r**3
c         write(6,*)'distancia r',r, rinv, r3
        fx(i) = fact*(- rx(i)*rz(i)/r3 )
        fy(i) = fact*(- ry(i)*rz(i)/r3 )
        fz(i) = fact*( rinv - rz(i)*rz(i)/r3 )
      enddo

c      write(6,*)'fuerzas'
c      write(6,*)'n',n
c      do i=1,n
c         write(6,*)'fuerza',i,fx(i),fy(i),fz(i)
c      enddo

      return
      end

      subroutine posiciones(nmax,n,amasa,dt,rx,ry,rz,fx,fy,fz,
     &dx,dy,dz,dif0)
      implicit double precision (a-h,o-z)
      integer is
      real*8 rx(nmax),ry(nmax),rz(nmax)
      real*8 fx(nmax),fy(nmax),fz(nmax)
      real*8 dx(nmax),dy(nmax),dz(nmax)
      real*8 gx(nmax),gy(nmax),gz(nmax)
      real*8 dxr(nmax),dyr(nmax),dzr(nmax)
      real*8 amasa(nmax)
      real*8 vxr(nmax),vyr(nmax),vzr(nmax)
      real*8 vx(nmax),vy(nmax),vz(nmax)
c      real*8 cc1,cc0,ran,cc10,cc11

c      write(6,*)'---En dinamicab---'
      do i=1,n
         dx(i) = 0.0d0
         dy(i) = 0.0d0
         dz(i) = 0.0d0
      enddo

c        velocidades iniciales
      do i=1,n
         vxr(i) = 0.0d0
         vyr(i) = 0.0d0
         vzr(i) = 0.0d0
      enddo

c         velocidades para la fuerza de fricción del medio
      do i=1,n
         call azarg(is,vx(i))
         call azarg(is,vy(i))
         call azarg(is,vz(i))

          vxr(i) = vx(i)*dsqrt(2.0d0*dif0*dt)
          vyr(i) = vy(i)*dsqrt(2.0d0*dif0*dt)
          vzr(i) = vz(i)*dsqrt(2.0d0*dif0*dt)
      enddo


      do i=1,n
         call azarg(is,gx(i))
         call azarg(is,gy(i))
         call azarg(is,gz(i))
c         write(17,'(i5,3f12.6)')i,gx(i),gy(i),gz(i)

c calculo del desplazamiento aleatorio
         dxr(i) = gx(i)*dsqrt(2.0d0*dif0*dt)
         dyr(i) = gy(i)*dsqrt(2.0d0*dif0*dt)
         dzr(i) = gz(i)*dsqrt(2.0d0*dif0*dt)

c con esta expresion para rx(i) incluimos desplazamiento debido al campo y aleatorio

         rx(i) = rx(i) + dt*dif0*fx(i)/amasa(i) + dxr(i) - indif0*vxr(i)*dt
         ry(i) = ry(i) + dt*dif0*fy(i)/amasa(i) + dyr(i) - indif0*vyr(i)*dt
         rz(i) = rz(i) + dt*dif0*fz(i)/amasa(i) + dzr(i) - indif0*vzr(i)*dt
c         write(6,*) i,rx(i),ry(i),rz(i)
c con esta expresion para rx(i) solo incluimos desplazamiento aleatorio 0.0005

c         rx(i) = rx(i) +  dxr(i)
c         ry(i) = ry(i) +  dyr(i)
c         rz(i) = rz(i) +  dzr(i)

c con esta expresion para rx(i) solo incluimos desplazamiento debido al campo

c         rx(i) = rx(i) + dt*dif0*fx(i)/amasa(i)
c         ry(i) = ry(i) + dt*dif0*fy(i)/amasa(i)
c         rz(i) = rz(i) + dt*dif0*fz(i)/amasa(i)
      enddo

      do i=1,n
        if(i.eq.18) then
          write(90,*) rx(i),ry(i),rz(i)
90         format(3F8.3)
        endif
      enddo


      return
      end

c revisar ecuacion de movimiento con factores correctos
c donde incluyamos el coeficiente de autodifusion en lugar de friccion

c---- incluir generador de numeros aleatorios para 
c     desplazamientos aleatorios dx(i)
c   * observar el comportamiento de las particulas en los 
c     casos a) campo, b) aleatorio, c) campo + aleatoria
c-- * localizar ecuacion para rotacion
c   * generar orientacionaciones tales que el vector sea unitario
c   * incluir orientacion de las partifculas en config inicial
c   * no colocar particulas en el origen para evitar divergencias en la fuerza
c   * modificar programa para que lea orientaciones
c   * generar desplazamientos orientacionales que satisfagan la 
c     condicion requerida en la orientacion inicial

c   * programar ecuacion de movimiento orientacional

c   * pensar en que propiedades podemos calcular
c   * incluir o no interacciones directas?
c   * hablar con Roberto

c--------------------------------------
      SUBROUTINE AZARG( ISEED,XX )
c--------------------------------------

      implicit double precision(a-h,o-z)
      integer NN1,ISEED
      real*8 PI,R,S,XX
      real*8 temp,cc11
      real*8 vx,vy,vz
c      real*8 vrx(nmax),vry(nmax),vrz(nmax)
c      real*8 vx(nmax),vy(nmax),vz(nmax)
c      real*8 cc1,cc0,ran,cc10,cc11
      PARAMETER (NN1=300)

c     ESTA SUBRUTINA produce numeros al azar
c     con DISTRIBUCION GAUSSIANA
c     REF: Allen-Tildeley

      PI=4.0d0*dATAN(1.0d0)
      temp=1.0d0
      CALL RANDOM(ISEED,R)
      CALL RANDOM(ISEED,S)

      XX=dSQRT(-2.0d0*temp*dLOG(1.0d0*R))*dcos(2.0d0*PI*S)

c      write(6,*) XX
      RETURN
      END
c--------------------------------------
c PARA CALCULAR NUMEROS ALEATORIOS
c--------------------------------------
      SUBROUTINE RANDOM( IDUM,RANDX )
c--------------------------------------

c     ESTA SUBRUTINA CALCULA NUMEROS AL AZAR
c     con distribucion uniforme, DADA LA SEMILLA IDUM
c     REF: Computers in Physics, Vol. 6 #5
c          sep-oct 1992, function ran2

      implicit double precision(a-h,o-z)
      INTEGER idum,im1,im2,imm1,ia1,ia2
      INTEGER iq1,iq2,ir1,ir2,ntab,ndiv
      real*8 am,eps,rnmx,randx
c      REAL am,eps,rnmx
      PARAMETER (IM1=2147483563, IM2=2147483399, AM=1.0D0/IM1
     &  ,IMM1=IM1-1, IA1=40014, IA2=40692, IQ1=53668
     &  ,IQ2=52774, IR1=12211, IR2=3791, NTAB=32
     &  ,NDIV=1+IMM1/NTAB, EPS=1.2d-7, RNMX=1.0d0-EPS)

       INTEGER idum2,j,k,IV(NTAB),iy
       SAVE IV,IY,IDUM2
       DATA IDUM2/123456789/, IV/NTAB*0/, IY/0/

       IF(IDUM .LE. 0.0d0) THEN
       IDUM=MAX(-IDUM,1)
       IDUM2=IDUM
       
       DO J=NTAB+8,1,-1
         K=IDUM/IQ1
         IDUM=IA1*(IDUM-K*IQ1)-K*IR1
         IF(IDUM .LT. 0.0d0) IDUM=IDUM+IM1
         IF(J.LE.NTAB) IV(J)=IDUM
       ENDDO
       IY=IV(1)
       ENDIF

       K=IDUM/IQ1
       IDUM=IA1*(IDUM-K*IQ1)-K*IR1
       IF(IDUM .LT. 0.0d0) IDUM=IDUM+IM1
       K=IDUM2/IQ2
       IDUM2=IA2*(IDUM2-K*IQ2)-K*IR2
       IF(IDUM2 .LT. 0.0d0) IDUM2=IDUM2+IM2
       J=1+IY/NDIV
       IY=IV(J)-IDUM2
       IV(J)=IDUM
       IF(IY .LT. 1.0d0) IY=IY+IMM1
       RANDX = MIN(AM*IY,RNMX)

      RETURN
      END  

      subroutine aleatorio(ra)
      implicit double precision(a-h,o-z)

      integer :: values(1:8), k
      integer, dimension(:), allocatable :: seed
      real(8) :: ra

      call date_and_time(values=values)

      call random_seed(size=k)
      allocate(seed(1:k))
      seed(:) = values(8)
      call random_seed(put=seed)
      call random_number(ra)
      write(6,*)ra
 
      return
      end

      
      subroutine gro(ipaso,nmax,n,rx,ry,rz,boxx,boxy,boxz,iesp)
      implicit double precision(a-h,o-z)
      integer iesp(nmax)
      real*8 rx(nmax),ry(nmax),rz(nmax)
      character*5 resid_name,symbol
      real*8 sr

c      sr = 0.3405d0
      sr = 1.0d0

      resid_name = 'SOL  '

      write(88,*) 'CONFIGURACION ', ipaso
      write(88,*) n

      do i=1,n
        if(i.eq.1) symbol = ' O  '
        write(88,10) i,resid_name,symbol,i,
     &  rx(i)*sr,ry(i)*sr,rz(i)*sr
        if(i.eq.1) write(39,'(I5,3F8.3)')ipaso,rx(i),ry(i),rz(i)
        write(88,*) rx(i),ry(i),rz(i)
      enddo

cc      write(88,'(3F12.5)') boxx*sr,boxy*sr,boxz*sr
      write(88,*) rx(i),ry(i),rz(i)

10    format(I5,2A5,I5,3F8.3)

      return
      end

      subroutine norma(nmax,n,ox,oy,oz,onorma)
      implicit double precision(a-h,o-z)
      real*8 ox(nmax),oy(nmax),oz(nmax)
      real*8 onorma(nmax)

c      write(6,*) 'norma de vectores de orientacion'
      do i=1,n
       onorma(i)=dsqrt(ox(i)**2 + oy(i)**2 + oz(i)**2)
c       write(6,*) i, onorma(i)
      enddo

      return
      end


      subroutine orientacionaleatoria(nmax,n,dt,oxr,oyr,ozr,ox,oy,oz,
     & campo,difr0,normaf,np,amasa,temp)
      implicit double precision (a-h,o-z)
      integer is
      real*8 tx(nmax), ty(nmax),tz(nmax)
      real*8 oxr(nmax),oyr(nmax),ozr(nmax)
      real*8 normarandom(nmax),normarandomw(nmax)
      real*8 ox(nmax),oy(nmax),oz(nmax)
      real*8 theta(nmax),amasa(nmax),phy(nmax)
      real*8 normaf(nmax),cosi(nmax),diffo(nmax)
      real*8 wx(nmax),wy(nmax), wz(nmax)
      real*8 wxr(nmax),wyr(nmax),wzr(nmax)
      real*8 p1(nmax),p2(nmax),p4(nmax)
      real*8 tcalx(nmax),tcaly(nmax),tcalz(nmax)
      real*8 c0x(nmax),c0y(nmax),c0z(nmax)
      real*8 normarandomc0(nmax)
      character*1 cc
      real*8 temp
c      write(6,*) 'Autodifusion rotacional',difr0,n
cc-----------------------------------------------------------FUERZA ALEATORIA PASA A VELOCIDAD
cc      do i=1,n
cc        call azarg(is,wx(i))
cc        call azarg(is,wy(i))
cc        call azarg(is,wz(i))

cc         wxr(i) = wx(i)*dsqrt(2.0d0*temp*temp*(1.0d0/difr0)*dt)
cc         wyr(i) = wy(i)*dsqrt(2.0d0*temp*temp*(1.0d0/difr0)*dt)
cc         wzr(i) = wz(i)*dsqrt(2.0d0*temp*temp*(1.0d0/difr0)*dt)
cc         normarandomw(i) = dsqrt(wxr(i)**2 + wyr(i)**2 + wzr(i)**2)
cc         wxr(i) = (wxr(i)/normarandomw(i))*(amasa(i)*dt)
cc         wyr(i) = (wyr(i)/normarandomw(i))*(amasa(i)*dt)
cc         wzr(i) = (wzr(i)/normarandomw(i))*(amasa(i)*dt)
c         write(6,*) i, wxr(i), wyr(i), wzr(i),normarandomw(i)
cc      enddo
cc-----------------------------------------------------------REESCALAMIENTO DE VELOCIDADES
      
cc      do i=1,n
cc         tcalx(i)=(1.d0/np)*wxr(i)**2 
cc         tcaly(i)=(1.d0/np)*wyr(i)**2 
cc         tcalz(i)=(1.d0/np)*wzr(i)**2 
c        write(6,*) i,np, tcalx(i),wxr(i),wyr(i),wzr(i)
cc         c0x(i)=dsqrt(temp/tcalx(i))
cc         c0y(i)=dsqrt(temp/tcaly(i))
cc         c0z(i)=dsqrt(temp/tcalz(i))
c         normarandomc0(i)= 1.0
c         c0x(i)=c0x(i)/normarandomc0(i)
c         c0y(i)=c0y(i)/normarandomc0(i)
c         c0z(i)=c0z(i)/normarandomc0(i)
c         write(6,*) i,np,c0x(i),c0y(i),c0z(i),normarandomc0(i)
cc      enddo
cccc está listo para los pasos

cc-----------------------------------------------------------DESPLAZAMIENTO ANGULAR
      do i=1,n
         call azarg(is,wx(i))
         call azarg(is,wy(i))
         call azarg(is,wz(i))
c         write(6,*) wx(i),wy(i),wz(i)

c         write(6,*) i, tox(i),toy(i),toz(i)
         normarandom(i) = dsqrt(tx(i)**2 + ty(i)**2 + tz(i)**2)
         oxr(i) = tx(i)/normarandom(i)
         oyr(i) = ty(i)/normarandom(i)
         ozr(i) = tz(i)/normarandom(i)

         oxr(i) = oxr(i)*dsqrt(2.0d0*difr0*dt)
         oyr(i) = oyr(i)*dsqrt(2.0d0*difr0*dt)
         ozr(i) = ozr(i)*dsqrt(2.0d0*difr0*dt)

cc        write(6,*) oxr(i),oyr(i),ozr(i)
cc        write(6,*)'o aleaoria',i,oxr(i),oyr(i),ozr(i),normarandom(i)

cc-----------------------------------------------------------DINAMICA LANGEVIN
         factor = campo*difr0
         ox(i) = ox(i) + 0.005*oxr(i) + factor*dt*(-ox(i)*oz(i)**2) 
         oy(i) = oy(i) + 0.005*oyr(i) + factor*dt*(-oy(i)*oz(i)**2) 
         oz(i) = oz(i) + 0.005*ozr(i) + factor*dt*(+oz(i)*ox(i)**2 
     &   +oz(i)*oy(i)**2)

         normaf(i) = dsqrt(ox(i)**2 + oy(i)**2 + oz(i)**2)
         ox(i) = ox(i)/normaf(i)
         oy(i) = oy(i)/normaf(i)
         oz(i) = oz(i)/normaf(i)
c         write(6,*) i, ox(i), oy(i), oz(i)

cc-------------------------------------------------------------CALCULO DEL ANGULO POLAR
	 if (oz(i).gt.0) then
	    theta(i)=datan(dsqrt(ox(i)**2 + oy(i)**2)/oz(i))
	 else if (oz(i).eq.0) then
	     theta(i)=pp/2
         else if(oz(i).lt.0) then
             theta(i)=pp + datan(dsqrt(ox(i)**2 + oy(i)**2)/oz(i))
         end if
	 cosi(i)=dcos(theta(i))


cc-------------------------------------------------------------CALCULO DEL ANGULO PHI

	 if (ox(i).gt.0) then
	    if (oy(i).gt.0) then
               phy(i)=datan(oy(i)/ox(i))
            else if(oy(i).lt.0) then
                 phy(i)=2*(4.0d0*dATAN(1.0d0)) + datan(oy(i)/ox(i))
           end if
	 else if (ox(i).eq.0) then
	     phy(i)=((4.0d0*dATAN(1.0d0))/2)*(oy(i)/dabs(oy(i)))
         else if(ox(i).lt.0) then
             phy(i)=4.0d0*dATAN(1.0d0) + datan(oy(i)/ox(i))
         end if

cc-----------------------------------------------------------ANALISIS POLINOMIOS DE LEGENDRE
	 p1(i)=cosi(i)
	 p2(i)=(3/2)*cosi(i)**2 - (1/2)
	 p4(i)=(35/8)*cosi(i)**4 - (30/8)*cosi(i)**2 + (3/8) 
	 diffo(i)= 1 + (5/14)*p2(i) + (8/7)*p4(i)
c        fcs(i)=1 + (5/7)*p2(i) - (12/7)*p4(i)
c        diffo(i)= 1 + (5/14)*p2(i) + (8/7)*p4(i)
cc         write(6,*) i,diffo(i),p1(i),p2(i),p4(i)
cc       write(6,*) oxr(i),oyr(i),ozr(i)
c        write(6,*)'nuevaorien',i,ox(i),oy(i),oz(i),normaf(i)
       enddo


       do i=1,n
         if(i.eq.18) then
c          write(6,'(f16.14,a,f16.14,a,f16.14,a)')ox(i),cc,oy(i),cc,oz(i),cc
cc            write(6,*) diffo(i),theta(i)
cc            write(6,*) diffo(i), theta(i)
cc           write(6,*)ox(i),cc,oy(i),cc,oz(i),cc
            write(86,*) ox(i),oy(i),oz(i)
86         format(3F8.3)
cc          write(6,*) n,cc,dt,cc,diffo
         endif
       enddo

       return
       end


