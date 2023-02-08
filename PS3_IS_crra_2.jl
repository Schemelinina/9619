## PS3
### Polynomial interpolation using monomial basis. I used examples from the book by Otken
#Define function and grid
#CRRA, sigma=2 function

f1(x)=x.^(1-2)/(1-2)
x=(range(0.05,2;length=5)); 
y1=map(f1,x)

#Build matrix A
A1 = zeros(Float64, length(x), length(x)-1)
for i in 1:length(x), j in 1:length(x)-1
    A1[i, j] = (x[i]^j) 
end

a=ones(length(x))
A=hcat(a,A1)

#Find vector of coefficients a for each function
a1=A\y1

using Plots
plot(x, y1, label = "crra,2, monomial")

#Check for accuracy

p1(x)=a1[1]+a1[2]x+a1[3]x^2+a1[4]x^3+a1[5]x^4
p_1=map(p1,x)
dif1=(p_1-y1)
mse1=sqrt(sum(dif1)^2/length(x))


#size 3 (i did it manually due to errors in loops for different n)
f2(x1)=x1.^(1-2)/(1-2)
x1=(range(0.05,2;length=3)); 
y2=map(f2,x1)

#Build matrix A
A2 = zeros(Float64, length(x1), length(x1)-1)
for i in 1:length(x1), j in 1:length(x1)-1
    A2[i, j] = (x1[i]^j) 
end

a=ones(length(x1))
B=hcat(a,A2)

#Find vector of coefficients a for each function
a2=B\y2

using Plots
plot!(x1, y2, label = "crra,2, monomial, diff sizes")

#Check for accuracy

p2(x1)=a2[1]+a2[2]x1+a2[3]x1^2
p_2=map(p2,x1)
dif2=(p_2-y2)
mse2=sqrt(sum(dif2)^2/length(x1))

#Extrapolation using monomial basis  

x_ext=(range(0.02,2.5;length=5)); 
p(x_ext)=a1[1]+a1[2]x_ext+a1[3]x_ext^2+a1[4]x_ext^3+a1[5]x_ext^4
using Plots
plot(p,x_ext, title="crra,2,monomial extrapolation")

f_ext(x_ext)=x_ext.^(1-2)/(1-2)
plot!(f_ext,x_ext)



##Polynomial interpolation using Newton basis 
#I used the code from Okten

using ForwardDiff
import Base.diff
import ForwardDiff

function diff(x::Array,y::Array)
    m = length(x) #m is the number of data points
    an = zeros(m)
    for i in 1:m
        an[i]=y[i]
    end
    for j in 2:m
        for i in reverse(collect(j:m))
            an[i]=(an[i]-an[i-1])/(x[i]-x[i-(j-1)])
        end
    end
    return(an)
end

diff(collect(x),y1)


function newton(x::Array,y::Array,z)
    m=length(x) #here m is the number of data points, not the degree # of the polynomial
    an=diff(x,y)
    sum=an[1]
    pr=1.0
    for j in 1:(m-1)
        pr=pr*(z-x[j])
        sum=sum+an[j+1]*pr
    end
    return sum
end

#I used a package to produce plots as my code for A matrix was producing errors 

using Interpolations
xi=collect(x)
interp=map(z->newton(xi,y1,z),x) 
plot(xi,interp, title="crra,2_Newton")
#Extrapolating
plot!(f_ext,x_ext)


#Cubic natural spline method

function spline(x::Array,y1::Array)
    m=length(x) # m is the number of data points
    n=m-1
    global as=Array{Float64}(undef,m)
    global b=Array{Float64}(undef,n)
    global c=Array{Float64}(undef,m)
    global d=Array{Float64}(undef,n)
    for i in 1:m
    as[i]=y1[i]
    end
    h=Array{Float64}(undef,n)
    for i in 1:n
    h[i]=x[i+1]-x[i]
    end
    u=Array{Float64}(undef,n)
    u[1]=0
    for i in 2:n
        u[i]=3*(as[i+1]-a[i])/h[i]-3*(as[i]-as[i-1])/h[i-1]
        end
        s=Array{Float64}(undef,m)
        z=Array{Float64}(undef,m)
        t=Array{Float64}(undef,n)
        s[1]=1
        z[1]=0
        t[1]=0
        for i in 2:n
        s[i]=2*(x[i+1]-x[i-1])-h[i-1]*t[i-1]
        t[i]=h[i]/s[i]
        z[i]=(u[i]-h[i-1]*z[i-1])/s[i]
        end
        s[m]=1
        z[m]=0
        c[m]=0
        for i in reverse(1:n)
        c[i]=z[i]-t[i]*c[i+1]
        b[i]=(as[i+1]-as[i])/h[i]-h[i]*(c[i+1]+2*c[i])/3
        d[i]=(c[i+1]-c[i])/(3*h[i])
        end
end
            
function spline_ev(w,x::Array)
    m=length(x)
    if w<x[1]||w>x[m]
    return print("error: spline evaluated outside its domain")
    end
    n=m-1
    p=1
    for i in 1:n
    if w<=x[i+1]
    break
    else
    p=p+1
    end
    end
    # p is the number of the subinterval w falls into, i.e., p=i
    # means w falls into the ith subinterval $(x_i,x_{i+1}),
    # and therefore the value of the spline at w is
    # a_i+b_i*(w-x_i)+c_i*(w-x_i)^2+d_i*(w-x_i)^3.
    return as[p]+b[p]*(w-x[p])+c[p]*(w-x[p])^2+d[p]*(w-x[p])^3
    end

xi=collect(x)
spline(xi,y1) 
naturalspline=map(z->spline_ev(z,xi),x)
plot(x,int,label="cubic spline, crra,2")

#Extrapolating
    xi_ext=collect(x_ext)
    y_ext=map(f1,x_ext)
    spline(xi_ext,y_ext) 
    naturalspline_ext=map(z->spline_ev(z,xi_ext),x_ext)
plot!(x_ext,naturalspline_ext,label="cubic spline_crra2_extrapolation")