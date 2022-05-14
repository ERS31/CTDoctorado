      program config0
      implicit none
      integer nmax,n,i,kk,l
      integer mm,nn,nx,ny,nz
      parameter(mm=10)
      parameter(nmax=mm**(3))
      real*8 pi,boxx,rho,dl,r3n
      real*8 rx(nmax),ry(nmax),rz(nmax)
      real*8 vx(nmax),vy(nmax),vz(nmax)
      real*8 th(nmax),ph(nmax)
      real*8 ox(nmax),oy(nmax),oz(nmax)
      real*8 onorma(nmax)

      open(12,file='1particula0.dat')

      pi = 4.0d0*datan(1.0d0)

      write(6,*)'Dar el numero l: n = l**3'
      read(5,*) l

      n = l**3

      write(6,*)'el numero de particulas es',n

      if(n.gt.nmax) then 
       write(6,*)'aumentar nmax'
       stop
      endif

      write(6,*)'Dar la densidad'
      read(5,*) rho

      boxx = (dble(n)/rho)**(1.0d0/3.0d0)
      write(6,*)'boxx = ',boxx

      if(l.gt.mm) then
       write(6,*)'aumentar mm'
       stop
      endif

c sc
      dl=boxx/dble(l)

      kk=0
      do nz=1,l
        do ny=1,l
          do nx=1,l
             i=(nx-1)+(ny-1)*l+(nz-1)*l**(2)+1
             write(6,*)'nx,ny,nz,i',nx,ny,nz,i
             rx(i)=(nx-1)*dl      
             ry(i)=(ny-1)*dl      
             rz(i)=(nz-1)*dl      
             th(i)=pi/3.0d0
             ph(i)=pi/4.0d0
             kk = kk + 1
           enddo
        enddo
      enddo

      write(6,*)'numero de particulas generadas',kk

      do i=1,kk
        ox(i)=dsin(th(i))*dcos(ph(i)) 
        oy(i)=dsin(th(i))*dsin(ph(i)) 
        oz(i)=dcos(th(i))
      enddo

      do i=1,kk
       vx(i)=0.0d0
       vy(i)=0.0d0
       vz(i)=0.0d0
      enddo

      write(6,*) 'norma de vectores de orientacion'
      do i=1,kk
       onorma(i)=dsqrt(ox(i)**2 + oy(i)**2 + oz(i)**2)
       write(6,*) i, onorma(i)
      enddo

      write(6,*)'escribiendo configuracion'

      write(12,*) kk
      write(12,*) boxx,boxx,boxx
      do i=1,kk
       write(12,'(i6,9f12.5)')i,rx(i),ry(i),rz(i),vx(i),vy(i),vz(i),
     & ox(i),oy(i),oz(i)
      enddo

      close(12)

      stop 
      end
