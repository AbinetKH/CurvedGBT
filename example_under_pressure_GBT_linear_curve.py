# -*- coding: utf-8 -*-
"""
Created on Mon Jul 06 08:35:32 2015
@author: Abinet Habtemariam
GBT linear FEM implementation 
"""
import time
import timeit
from mpl_toolkits.mplot3d import Axes3D
from GBT_classes_list import *    
from matplotlib import colors as mcolors
import sys
#getcontext().prec = 30         
"""Define/initialize variables"""
num_modes=50
type_of_load='pressure'
q=10#66/1000/1000

no_beam_elem=80#10 #
varphi_total=math.pi/4
cross_sect_ref=189
list_of_beam_elem=np.array([40]) # to extract section detail
varphi=varphi_total/no_beam_elem
E=float(205000) # 100.0#689.475908677537#
t=float(15)
r=float(300)
R=float(3000)
length=float(varphi_total*R)#10.0#254#
l=length/no_beam_elem
mu=0.3
xi=1.0
#q=-50000/(2*math.pi*r)
calculate_internal_forces=1# turn on=1 or off=0
plot_3d=1 # turn on=1 or off=0
gravity=9810
density=7.83e-9
K_spring=0.0
start_time = time.time()

""" load participation"""  

[mat_csi_p,mode_list_p]=Load_shared_mode.load_contr(l,r, t, 0, (900*math.pi/1800) ,(2700*math.pi/1800), num_modes, type_of_load,gravity,density,xi,R)
[mat_csi_m,mode_list_m]=Load_shared_mode.load_contr(l,r, t, 0, (2700*math.pi/1800),(4500*math.pi/1800), num_modes, type_of_load,gravity,density,xi,R)

[mat_csi_p,mode_list_p]=Load_shared_mode.load_contr(l,r, t, q, (0*math.pi/1800) ,(3600*math.pi/1800), num_modes, type_of_load,gravity,density,xi,R)

"""All_modes"""
#mode_list=['t|t','a|a','1|1','2|c','2|v','2|u','3|c','3|v','3|u','4|c','4|v','4|u','5|c','5|v','5|u','6|c','6|v','6|u','7|c','7|v','7|u','8|c','8|v','8|u','9|c','9|v','9|u','10|c','10|v','10|u']#,'11|c','11|v','11|u','9|c','9|v','9|u','11|c','11|v','11|u','15|c','15|v','15|u']
"""Bernoulli_beam"""
#mode_list=['1|1','3|c']
"""Timoshenko_beam"""
#mode_list=['1|1','3|c','3|v','3|u']
"""Transverse only"""
#mode_list=['a|a','5|c','5|v','5|u']
"""classical_modes"""
#mode_list=['a|a','1|1','2|c','3|c','4|c','5|c','6|c','7|c','8|c','9|c','10|c','11|c']#]
#mode_list=['1|1','3|c','5|c','7|c','9|c','11|c']
#mode_list=['a|a','1|1','3|c','3|v','5|c','5|v','7|c','7|v','9|c','9|v','11|c','11|v','13|c','13|v']#]
#mode_list=['a|a','1|1','3|c','3|u','5|c','5|u','7|c','7|u','9|c','9|u','11|c','11|u','13|c','13|u']
mode_list=['a|a','1|1','3|c','3|v','3|u','5|c','5|v','5|u','7|c','7|v','7|u']#,'9|c','9|v','9|u','11|c','11|v','11|u','13|c','13|v','13|u']#,'15|c','15|v','15|u','17|c','17|v','17|u','19|c','19|v','19|u','23|c','23|v','23|u','27|c','27|v','27|u']#],'9|c','11|c','13|c','15|c']
#mode_list=['t|t','a|a','1|1','2|c','2|v','2|u','3|c','3|v','3|u','4|c','4|v','4|u','5|c','5|v','5|u','6|c','6|v','6|u','7|c','7|v','7|u','8|c','8|v','8|u','9|c','9|v','9|u','10|c','10|v','10|u']
#mode_list=['a|a','1|1','3|c','3|v','5|c','5|v','7|c','7|v','9|c','9|v','11|c','11|v','15|c','15|v']#,'9|c','11|c','13|c','15|c']

"""random_modes"""
#mode_list=['a|a','1|1','3|c','3|v','3|u','5|c','5|v','5|u','7|c','7|v','7|u','11|c','11|v','11|u']

mat_csi=mat_csi_p
mat_csi[:,2]=mat_csi[:,2] + mat_csi_m[:,2]
mat_csi[:,3]=mat_csi[:,3] + mat_csi_m[:,3]
mode_list_size=len(mode_list)


#spring_stiffness=spring_stiffness_GBT_mode.sprGBT(spring_stiffness_GBT_mode(K_spring,r,(-9*math.pi/18),(9*math.pi/18),num_modes))


"""   X tensor to vector      """
LM_element=LM_connectivity_array(mode_list,1)
LM_vector=LM_element.LM_array()
ik_vector_all=np.zeros((mode_list_size**2,20),object)
count=0
x_tol=1e-30
x_tol_s=1e-30
for a in range(mode_list_size):
    for b in range(mode_list_size):
            k1=mode_list[a]
            k2=mode_list[b]
            Linear_ik=linear_curved_GBT_ik_coupling_matrix(k1,k2,R,t,r,xi,mu)
            C1=Linear_ik.C1_ik()
            C2=Linear_ik.C2_ik()
            C3=Linear_ik.C3_ik()
            C4=Linear_ik.C4_ik()
            B_t=Linear_ik.B_ik()
            G_s=Linear_ik.D_ik()
            D_1mu=Linear_ik.D_1mu_ik()
            D_2mu=Linear_ik.D_2mu_ik()
            D_3mu=Linear_ik.D_3mu_ik()
            D_4mu=Linear_ik.D_4mu_ik()
            
            if  abs(C1 or C2 or C3 or C4 or B_t or G_s or D_1mu or D_2mu or D_3mu or D_4mu)>x_tol:
                lm_a=a*4
                lm_b=b*4
                ik_vector_all[count,0:2]=[k1,k2]
                ik_vector_all[count,2:12]=[C1,C2,C3,C4,B_t,G_s,D_1mu,D_2mu,D_3mu,D_4mu]              
                ik_vector_all[count,12:16]=LM_vector[lm_a:lm_a+4,0]                
                ik_vector_all[count,16:20]=LM_vector[lm_b:lm_b+4,0]                                                  
                count+=1
ik_vector=ik_vector_all[0:count,:]
count=0

# ik_vector_all_additional=np.zeros((mode_list_size**2,20),object)
# count=0
# x_tol=1e-30
# x_tol_s=1e-30
# for a in range(mode_list_size):
#     for b in range(mode_list_size):
#             k1=mode_list[a]
#             k2=mode_list[b]
#             Linear_ik=linear_curved_GBT_ik_coupling_matrix_additional(k1,k2,R,t,r,xi,mu)
#             C1=Linear_ik.C1_ik()
#             C2=Linear_ik.C2_ik()
#             C3=Linear_ik.C3_ik()
#             C4=Linear_ik.C4_ik()
#             B_t=Linear_ik.B_ik()
#             G_s=Linear_ik.D_ik()
#             D_1mu=Linear_ik.D_1mu_ik()
#             D_2mu=Linear_ik.D_2mu_ik()
#             D_3mu=Linear_ik.D_3mu_ik()
#             D_4mu=Linear_ik.D_4mu_ik()
            
#             if abs(C1)>x_tol or abs(C2)>x_tol or abs(C3)>x_tol or abs(C4)>x_tol or abs(B_t)>x_tol or abs(G_s)>x_tol:
#                 lm_a=a*4
#                 lm_b=b*4
#                 ik_vector_all_additional[count,0:2]=[k1,k2]
#                 ik_vector_all_additional[count,2:12]=[C1,C2,C3,C4,B_t,G_s,D_1mu,D_2mu,D_3mu,D_4mu]              
#                 ik_vector_all_additional[count,12:16]=LM_vector[lm_a:lm_a+4,0]                
#                 ik_vector_all_additional[count,16:20]=LM_vector[lm_b:lm_b+4,0]                                                  
#                 count+=1
# ik_vector_additional=ik_vector_all_additional[0:count,:]

"""   Boolean array or LM_marix size element dof X number of element  #assembly      """
LM_inst=LM_connectivity_array(mode_list,no_beam_elem)
LM_matrix=LM_inst.LM_array()

""" Define zero vectors """
V=np.zeros((no_beam_elem+1,mode_list_size+1),float)
V2=np.zeros((no_beam_elem+1,mode_list_size+1),float)
beta_o=np.zeros((no_beam_elem),float) # the vector of initial angles of beam members, measured with respect to the global X axis
beta=np.zeros((no_beam_elem),float)
u_previous=np.zeros((Dof.Global(Dof(mode_list,mat_csi,no_beam_elem))),float) #the vector of global nodal displacements for the whole structure being analyzed, initially u = 0
stiffness_matrix_global=np.zeros((len(u_previous),len(u_previous)),float)
F_ext=np.zeros((len(u_previous)),float)
f_internal_global=np.zeros((len(u_previous)),float)
F_previous=np.zeros((len(u_previous)),float)
L_cur=np.zeros((no_beam_elem),float)+l # the vector of beam element lengths based on current u using equation (2)
Lo=np.zeros((no_beam_elem),float)+l # save the initial lengths in a vector Lo by using equation (1)


"""gbt loading"""
nodal_load=loading_cuved_local_w_loading(R,r,varphi,mode_list,q,LM_vector,0,math.pi*2,xi)
for i in range(0,no_beam_elem):      
    F_ext[LM_matrix[:,i]]+=nodal_load.element()

    ext_ele=nodal_load.element()
##########################
""" DOF """
torsion_dof=['fix','fix']
axisymmetric_dof=['fix','fix']
extension_dof=['fix','fix']
bending_dof=['fix','fix']
allocal_dof=['fix','fix']
[store_delet_row,store_insert_row]=support.dof(support(mode_list,mat_csi,no_beam_elem,torsion_dof,axisymmetric_dof,extension_dof,bending_dof,allocal_dof,LM_matrix))

""" mesh generator """
# u = np.linspace(0,2*math.pi,cross_sect_ref)
# u_shell= np.linspace(0,2*math.pi,36)
# rr=r*np.ones((no_beam_elem+1,1))
# w=np.linspace(0,length,no_beam_elem+1)
# x = rr* np.cos(u)
# y = rr* np.sin(u)
# z=np.ones((cross_sect_ref,1))*w


""" mesh generator """

r_d = np.linspace(0,2*math.pi,cross_sect_ref)
u_shell = r_d[0:len(r_d)-1]
l_d = np.linspace(0,varphi_total,no_beam_elem+1)
x=np.zeros((len(l_d),len(r_d)),float)
y=np.zeros((len(l_d),len(r_d)),float)
z=np.zeros((len(l_d),len(r_d)),float)
for i in range(len(l_d)):
  for j in range(len(r_d)):
      x[i,j] = (r * np.cos(r_d[j])+ R )* np.sin(l_d[i])
      z[i,j] = r * np.sin(r_d[j])#+ R * nm.cos(uR[i])
      y[i,j] =(r * np.cos(r_d[j])+ R ) * np.cos(l_d[i])#+ + (x[i,j] * nm.sin(i) + y[i,j] * nm.sin(i)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# surf = ax.plot_surface(x,y,z, rstride=1, cstride=1,color='g')
# ax.set_xlabel('X axis')
# ax.set_ylabel('Y axis')
# ax.set_zlabel('Height')
# ax.auto_scale_xyz([-R, R], [-R, R], [-R, R])
# ax.view_init(elev=0,azim=0)
# xlm=ax.get_xlim3d() #These are two tupples
# ylm=ax.get_ylim3d() #we use them in the next
# zlm=ax.get_zlim3d() #graph to reproduce the magnification from mousing
# ax.set_xlim3d(xlm[0],xlm[1])  
# ax.set_ylim3d(ylm[0],ylm[1])
# ax.set_zlim3d(zlm[0],zlm[1]) 




"""Stiffness_matrix_on_off"""
k_e=1
"""internal force on_off"""
int_e=1

"""number of steps and iterations"""
ninc=1
no_iter=0
tol=0.01
tol2=0.5*10e100

#n,m,e,i,j,k,l
store_k=np.zeros((ninc,2),float)

x_store=np.zeros((no_beam_elem+1,ninc),float) 
y_store=np.zeros((no_beam_elem+1,ninc-1),float) 

# x_data_3d=np.zeros((ninc+1,no_beam_elem+1,cross_sect_ref),float)
# y_data_3d=np.zeros((ninc+1,no_beam_elem+1,cross_sect_ref),float)
# z_data_3d=np.zeros((ninc+1,no_beam_elem+1,cross_sect_ref),float)

# x_data_3d[0,:,:]=x
# y_data_3d[0,:,:]=y
# z_data_3d[0,:,:]=z

V_store=np.zeros((ninc,no_beam_elem+1,mode_list_size+1),float)
V2_store=np.zeros((ninc,no_beam_elem+1,mode_list_size+1),float)




""" elastic stiffness matrix"""
k_elastic_main=Elastic_stiffness_matrix_GBT_curved.element(Elastic_stiffness_matrix_GBT_curved(varphi,E,R,mode_list,mat_csi,ik_vector,mu,k_e))

# k_elastic_additional=Elastic_stiffness_matrix_GBT_curved_additional.element(Elastic_stiffness_matrix_GBT_curved_additional(varphi,E,R,mode_list,mat_csi,ik_vector_additional,mu,k_e))
#k_elastic2= Elastic_stiffness_matrix_GBT.element(Elastic_stiffness_matrix_GBT( L_cur[0],Lo[0],r,t,E,spring_stiffness,mu,mode_list,mat_csi,k_e))
k_elastic=k_elastic_main#+k_elastic_additional
print ('k_elastic' )



""" start loop """
count=0
count_s=0
for i in range(0,ninc):
    print (i)
    stiffness_matrix_global_el=np.zeros((len(u_previous),len(u_previous)),float)
    u_element=np.zeros((Dof.Element(Dof(mode_list,mat_csi,no_beam_elem)) ),float)     
    f_internal_ele=np.zeros((Dof.Element(Dof(mode_list,mat_csi,no_beam_elem)) ),float) 
    for j in range(no_beam_elem):
        stiffness_matrix_global_el[np.transpose(np.tile(LM_matrix[:,j], (len(u_element), 1)).T),np.tile(LM_matrix[:,j], (len(u_element), 1)).T]=stiffness_matrix_global_el[np.transpose(np.tile(LM_matrix[:,j], (len(u_element), 1)).T),np.tile(LM_matrix[:,j], (len(u_element), 1)).T]+(k_elastic)

#    R_global_fix=np.delete((F_ext*(i+1)-f_internal_global),store_delet_row,axis=0) 
    penality_value=abs(stiffness_matrix_global_el.max())**4#vr
    stiffness_matrix_global_fix_el=stiffness_matrix_global_el[:,:]
    stiffness_matrix_global_fix_el[store_delet_row,store_delet_row]=penality_value
    u_current=np.linalg.solve(stiffness_matrix_global_fix_el,F_ext)
    
    # stiffness_matrix_global_fix_el=np.delete(stiffness_matrix_global_el,store_delet_row,axis=0)
    # stiffness_matrix_global_fix_el=np.delete( stiffness_matrix_global_fix_el,store_delet_row,axis=1)
            
    # u_incr_global=np.insert(u_fix,store_insert_row, 0,axis=0)
    
    # u_current=u_previous+u_incr_global
    
    # f_internal_global=np.zeros((len(u_previous)),float)
    # for j in range(0,no_beam_elem):
    #     u_element[:]=u_current[LM_matrix[:,j]]
    #   #  print u_element[:]
    #     f_ela_ele=Elastic_internal_force_GBT.element(Elastic_internal_force_GBT(L_cur[j],Lo[j],r,t,E,spring_stiffness,u_element,mu,mode_list,mat_csi,int_e))
        
         
    #     f_internal_global[LM_matrix[:,j]]=f_internal_global[LM_matrix[:,j]]+ (f_ela_ele)
      
    # R=F_ext*(i+1)-f_internal_global
    
    # R_step=F_ext*(i+1)-f_internal_global
    # R_fix=np.delete(R,store_delet_row,axis=0)
    

    u_previous[:]=u_current[:]
   
#    incre2=0
#    x2=np.zeros((no_beam_elem+1),float) # the vector of nodal x coordinates in the undeformed configuration
#    y2=np.zeros((no_beam_elem+1),float) # the vector of nodal y coordinates in the undeformed configuration
#    for j in range(0,no_beam_elem+1):
#        x2[j]=l*j
#    for j in range(0,no_beam_elem+1):
#        x2[j]+=u_previous[j]
#  #      x[j]+=u_previous[incre]#+=u_delta[incre]+u_incr_global[incre]
#        y2[j]+=u_previous[no_beam_elem+1+incre2]
# #       y[j]+=u_previous[1+incre]#+=u_delta[1+incre]+u_incr_global[1+incre]
#        incre2 +=2
#   # store_y[:,i]=x2
#    plt.figure(1)
#    plt.plot(x2, y2)
#    plt.show()
#plt.savefig('TL_gbt.png', format='png', dpi=600)
    """ calculation of amplification factor for each selected node """       
    [V,V2]=amplification.VV2(amplification(mode_list,mat_csi,no_beam_elem,u_previous,Lo[j]))
    print ('plot')

    if calculate_internal_forces==1:
    #linear    
    #longtudinal q=1,L=1000,r=500,t=10 fix bottom 
        # read ansys results
        for sect_no in range(len(list_of_beam_elem)):
            text_element_shell_data=open("%s" %('Ansys_shell_FEM_results_to_compare\element_section_forces_at_node_'+str(int(list_of_beam_elem[sect_no]))+'.txt'))
            text_element_shell_data_lines = text_element_shell_data.readlines()
            shell_ele_no=len(text_element_shell_data_lines)-1 #no_of_shell_ele_per_section
            shell_element_data=np.zeros((shell_ele_no,9),float)
            for i_line in range(0,shell_ele_no):
                for row_j in range(0,9): #columns are #ELEM  no.   N11 , N22 , N12 ,  M11 , M22 , M12,  Q13 , Q23
                    shell_element_data[i_line,row_j]=float((text_element_shell_data_lines[i_line+1].split( ))[row_j])
          #  list_of_beam_elem=np.array([9,89])       
            section_forces=Gbt_internal_forces_curved(mode_list,int(list_of_beam_elem[sect_no])-1,cross_sect_ref,u_previous,l,E,t,r,mu,R,xi,LM_matrix,varphi)
            N_x=section_forces.N_x()
            N_theta=section_forces.N_theta()
            N_x_theta=section_forces.N_x_theta()
            M_x=section_forces.M_x()
            M_theta=section_forces.M_theta()
            M_x_theta=section_forces.M_x_theta()
            Q_x=section_forces.Q_x()
            Q_theta=section_forces.Q_theta()
            
            N_x_plt=plot_internal_forces.plt(plot_internal_forces(r_d,shell_ele_no,0,N_x[:],shell_element_data[:,2],0,r, 300,"%s" %('N_x_at_'+str(int(list_of_beam_elem[sect_no]))+'N_per_mm')))
            N_theta_plt=plot_internal_forces.plt(plot_internal_forces(r_d,shell_ele_no,0,N_theta[:],shell_element_data[:,1],0,r, 300,"%s" %('N_theta_at_'+str(int(list_of_beam_elem[sect_no]))+'N_per_mm')))
            N_x_theta_plt=plot_internal_forces.plt(plot_internal_forces(r_d,shell_ele_no,0,N_x_theta[:],shell_element_data[:,3],0,r, 300,"%s" %('N_x_theta_at_'+str(int(list_of_beam_elem[sect_no]))+'N_per_mm')))    
        
            M_x_plt=plot_internal_forces.plt(plot_internal_forces(r_d,shell_ele_no,0,M_x[:],shell_element_data[:,5],0,r, 200,"%s" %('M_x_at_'+str(int(list_of_beam_elem[sect_no]))+'Nmm_per_mm')))
            M_theta_plt=plot_internal_forces.plt(plot_internal_forces(r_d,shell_ele_no,0,M_theta[:],shell_element_data[:,4],0,r, 300,"%s" %('M_theta_at_'+str(int(list_of_beam_elem[sect_no]))+'Nmm_per_mm')))
            M_x_theta_plt=plot_internal_forces.plt(plot_internal_forces(r_d,shell_ele_no,0,M_x_theta[:],shell_element_data[:,6],0,r, 300,"%s" %('M_x_theta_at_'+str(int(list_of_beam_elem[sect_no]))+'Nmm_per_mm')))    
        
        
            Q_x_plt=plot_internal_forces.plt(plot_internal_forces(r_d,shell_ele_no,0,Q_x[:],shell_element_data[:,8],0,r, 300,"%s" %('Q_x_at_'+str(int(list_of_beam_elem[sect_no]))+'N_per_mm')))
            Q_theta_plt=plot_internal_forces.plt(plot_internal_forces(r_d,shell_ele_no,0,Q_theta[:],shell_element_data[:,7],0,r, 300,"%s" %('Q_theta_at_'+str(int(list_of_beam_elem[sect_no]))+'N_per_mm')))

# ###############################################################################################################################    
    V_store[i,:,:]=V
    V2_store[i,:,:]=V2
     
    """ Calculation of displacements by combining amplification factors """
    #u displacment_local
    u_x=np.zeros((no_beam_elem+1,cross_sect_ref),float)
    v_y=np.zeros((no_beam_elem+1,cross_sect_ref),float)
    w_z=np.zeros((no_beam_elem+1,cross_sect_ref),float)
    w_z_l=np.zeros((no_beam_elem+1,cross_sect_ref),float)
    w_z_g=np.zeros((no_beam_elem+1,cross_sect_ref),float)

    
    v_y_big=np.zeros((no_beam_elem+1,cross_sect_ref),float)

    u_x123=np.zeros((no_beam_elem+1,cross_sect_ref),float)
    v_y123=np.zeros((no_beam_elem+1,cross_sect_ref),float)
    w_z123=np.zeros((no_beam_elem+1,cross_sect_ref),float)
    
    
    v_y_rotation=np.zeros((no_beam_elem+1,cross_sect_ref),float)
    #rect. system
    u_X=np.zeros((no_beam_elem+1,cross_sect_ref),float)
    w_Z=np.zeros((no_beam_elem+1,cross_sect_ref),float)
    v_Y=np.zeros((no_beam_elem+1,cross_sect_ref),float)

    varphi_add=0
    for a in range (0,no_beam_elem+1):
       vertheta=0#-2*math.pi/(cross_sect_ref-1)
       
       for j in range (0,cross_sect_ref):          
           sum_u=0
           sum_v=0
           sum_w=0
           local_sum_w=0
           global_sum_w=0.0
           sum_v123=0
           sum_w123=0
           for ia in range(mode_list_size):
               k=mode_list[ia]
               GBT=GBT_func_numpy_curve(k,r,vertheta,R)
           #     if k.split("|")[0]=='1':
           #         sum_u+=V2[a,ia+1]*GBT.warping_fun_u()
                 
           #     # # elif k.split("|")[0]=='2' or k.split("|")[0]== '3':
           #     # #    sum_v123+=V[a,ia+1]*GBT.warping_fun_v()
           #     # #    sum_w123+=V[a,ia+1]*GBT.warping_fun_w()
           #     else:
           # #    sum_u+=V2[a,ia+1]*GBT.warping_fun_u() 
           #         sum_v+=V[a,ia+1]*GBT.warping_fun_v()
           #         sum_w+=V[a,ia+1]*GBT.warping_fun_w() 
               sum_u+=V2[a,ia+1]*GBT.warping_fun_u()
               sum_v+=V[a,ia+1]*GBT.warping_fun_v()
               sum_w+=V[a,ia+1]*GBT.warping_fun_w() 
               if k.split("|")[0]!='1' and k.split("|")[0]!='2' and k.split("|")[0]!= '3':
                  local_sum_w+=V[a,ia+1]*GBT.warping_fun_w() 
               if k.split("|")[0]=='1' or k.split("|")[0]=='2' or k.split("|")[0]== '3':
                  global_sum_w+=V[a,ia+1]*GBT.warping_fun_w()                   
           u_x[a,j]=sum_u #+ V2[a,0]
           v_y[a,j]=sum_v
           w_z[a,j]=sum_w
           w_z_l[a,j]=local_sum_w
           w_z_g[a,j]=global_sum_w
           # v_y123[a,j]=sum_v123
           # w_z123[a,j]=sum_w123           
           """transfromation to global. system"""
           s_transf=Spherical_transformation_matrix(vertheta,varphi_add)
           XYZ=np.dot(s_transf.local_to_global(),[v_y[a,j],u_x[a,j],w_z[a,j]])
           u_X[a,j]=XYZ[0,1]
           v_Y[a,j]=XYZ[0,0]
           w_Z[a,j]=XYZ[0,2]
             #  if v is large 
           # v_y_big[a,j]=(sum_v/(r))+vertheta
           # v_y_rotation[a,j]=(sum_v/(r))
           # w_z_r[a,j]=(r+w_z[a,j])*np.cos(v_y_big[a,j])+(w_z123[a,j]*np.cos(vertheta)-v_y123[a,j]*np.sin(vertheta))
           # v_y_r[a,j]=(r+w_z[a,j])*np.sin(v_y_big[a,j])+(w_z123[a,j]*np.sin(vertheta)+v_y123[a,j]*np.cos(vertheta)) 

           
           vertheta+=(2*math.pi/(cross_sect_ref-1)) 
      # print(vertheta)    
       varphi_add+=varphi
    #  x[i,j] = (r * np.cos(theta)+ R )* np.sin(varphi)
    #  z[i,j] =  r * np.sin(theta)
    #  y[i,j] = (r * np.cos(theta)+ R ) * np.cos(varphi)
       
    print ('xyz')
    
    x_data=v_Y+x
    y_data=u_X+y
    z_data=w_Z+z
    
    #        # w_z_r[a,j]=w_z[a,j]*nm.cos(vertheta)-v_y[a,j]*nm.sin(vertheta)
    #        # v_y_r[a,j]=w_z[a,j]*nm.sin(vertheta)+v_y[a,j]*nm.cos(vertheta)           

    # x_data=nm.add(v_y_r,y)
    # y_data=nm.add(w_z_r,x)
    # z_data=u_x    

    """plot"""
    
    ###########################################################################################   
    for sect_no in range(len(list_of_beam_elem)):
        text_node_shell_data=open("%s" %('Ansys_shell_FEM_results_to_compare\pipe_deformation_data_at_node_'+str(int(list_of_beam_elem[sect_no]))+'.txt'))
        text_node_shell_data_lines = text_node_shell_data.readlines()
        shell_node_no=(len(text_node_shell_data_lines)-1)#no_of_shell_ele_per_section
        shell_node_data=np.zeros((shell_node_no,16),float)
        for i_line in range(0,shell_node_no):
            for row_j in range(0,7): #columns are #node no. ux uy uz X Y Z disp and coordinates 
               shell_node_data[i_line,row_j]=float((text_node_shell_data_lines[i_line+1].split( ))[row_j]) 
        varphi_section=varphi*(list_of_beam_elem[sect_no]-1)
        s_transf=Spherical_transformation_matrix(0,varphi_section)
        Guvw=np.dot(np.transpose(s_transf.local_to_global()),[np.sum (shell_node_data[:,3])/ shell_node_no,np.sum (shell_node_data[:,1])/ shell_node_no,np.sum (shell_node_data[:,2])/ shell_node_no]) 
        
        
        vertheta=0
        for i_transv in range(shell_node_no):
            s_transf=Spherical_transformation_matrix(vertheta,varphi_section)
            shell_node_data[i_transv,7:10]=np.dot(np.transpose(s_transf.local_to_global()),[shell_node_data[i_transv,3],shell_node_data[i_transv,1],shell_node_data[i_transv,2]]) 
            shell_node_data[i_transv,10:13]=[Guvw[:,0]*np.sin(vertheta),Guvw[:,1],Guvw[:,2]*np.cos(vertheta)]
            
        #    shell_node_data[i_transv,14]=np.sqrt(shell_node_data[i_transv,7]**2+shell_node_data[i_transv,8]**2+shell_node_data[i_transv,9]**2)
            vertheta+=(2*math.pi/(shell_node_no))
        shell_node_data[:,13:16]=shell_node_data[:,7:10]-shell_node_data[:,10:13]
        
        dispplot=plot_internal_forces.plt(plot_internal_forces(r_d,shell_node_no,0,w_z_l[int(list_of_beam_elem[sect_no])-1,:],shell_node_data[:,15],0,r, 300,'w__only_local_at_'+str(int(list_of_beam_elem[sect_no]))+'mm'))
 #       dispplot=plot_internal_forces.plt(plot_internal_forces(r_d,shell_node_no,0,w_z_l[int(list_of_beam_elem[sect_no]),:],shell_node_data[:,9],0,r, 1000,'u mm'))
        dispplot=plot_internal_forces.plt(plot_internal_forces(r_d,shell_node_no,0,w_z[int(list_of_beam_elem[sect_no])-1,:],shell_node_data[:,9],0,r, 300,'w_total_at_'+str(int(list_of_beam_elem[sect_no]))+' mm'))

        dispplot=plot_internal_forces.plt(plot_internal_forces(r_d,shell_node_no,0,w_z_g[int(list_of_beam_elem[sect_no])-1,:],shell_node_data[:,12],0,r, 200,'w_only_global_at_'+str(int(list_of_beam_elem[sect_no]))+' mm'))


        print (max(shell_node_data[:,9]))
        print (max(w_z[int(list_of_beam_elem[sect_no])-1,:]))
        print (np.mean(abs(100*(w_z[int(list_of_beam_elem[sect_no])-1,0:shell_node_no]-shell_node_data[:,9])/shell_node_data[:,9])))
 
       #  scale_disp=1
       #  x_shell_scaled=scale_disp*shell_node_data[:,1]+shell_node_data[:,4]
       #  y_shell_scaled=scale_disp*shell_node_data[:,2]+shell_node_data[:,5]
       #  y_data_scaled=((y_data[int(list_of_beam_elem[sect_no]),:])-x[int(list_of_beam_elem[sect_no]),:])*scale_disp+x[int(list_of_beam_elem[sect_no]),:]
       #  x_data_scaled=((x_data[int(list_of_beam_elem[sect_no]),:])-y[int(list_of_beam_elem[sect_no]),:])*scale_disp+y[int(list_of_beam_elem[sect_no]),:]        
       #  fig = plt.figure(figsize=(5,5))
       #  pos_signal = y_data_scaled.copy()
       #  neg_signal = y_data_scaled.copy()
       #  pos_signal[pos_signal <= 0] = np.nan
       #  neg_signal[neg_signal >= 0] = np.nan   
       # # plt.plot(y_data_scaled, x_data_scaled,  c= 'r',linewidth=2)    
       #  plt.plot(x_shell_scaled[:],y_shell_scaled[:],  'x', c= 'k',linewidth=8,markersize=12)
       #  plt.plot(y[int(list_of_beam_elem[sect_no]),:], x[int(list_of_beam_elem[sect_no]),:], 'k--',linewidth=1,markersize=10)
       #  ylist=y_data[int(list_of_beam_elem[sect_no]),:]-x[int(list_of_beam_elem[sect_no]),:]
       #  # print(ylist)
       #  maxy=np.max(ylist)
       #  plt.text(0,0,' Max= %5.2f' % float(maxy),{'color': 'r', 'fontsize': 15,'weight':'bold'})
       #  plt.text(50,100,' Min= %5.2f' % (float(-np.sum(y_data[int(list_of_beam_elem[sect_no]),:]-x[int(list_of_beam_elem[sect_no]),:])/len(x[int(list_of_beam_elem[sect_no]),:]))),{'color': 'b', 'fontsize': 15,'weight':'bold'})
       #  # plt.text(0,0,' Max= %5.2f' % (float(np.sum(x_data[int(list_of_beam_elem[sect_no]),:]-y[int(list_of_beam_elem[sect_no]),:])/len(y[int(list_of_beam_elem[sect_no]),:]))),{'color': 'r', 'fontsize': 15,'weight':'bold'})
       #  # plt.text(50,100,' Min= %5.2f' % (float(-np.sum(x_data[int(list_of_beam_elem[sect_no]),:]-y[int(list_of_beam_elem[sect_no]),:])/len(y[int(list_of_beam_elem[sect_no]),:]))),{'color': 'b', 'fontsize': 15,'weight':'bold'})
    
       #  plt.plot( x_data_scaled,pos_signal,  c= 'r',linewidth=2)
       #  plt.plot( x_data_scaled,neg_signal,  c= 'b',linewidth=2) 
       #  plt.fill_between(y[int(list_of_beam_elem[sect_no]),:],pos_signal ,x[int(list_of_beam_elem[sect_no]),:], color="none", hatch='//', edgecolor="r", linewidth=0.0)          
       #  plt.fill_between(y[int(list_of_beam_elem[sect_no]),:],neg_signal,x[int(list_of_beam_elem[sect_no]),:],  color="none", hatch='//', edgecolor="b", linewidth=0.0)  
       #  plt.axis('equal')
       #  plt.axis('off')
 
      #  plt.savefig('section3_{}.pdf'.format(i) , dpi=100, format='pdf', bbox_inches='tight')

    ###########################################################################################   
    if plot_3d==1:
        text_node_shell_data=open("%s" %('Ansys_shell_FEM_results_to_compare\pipe_deformation_data_at_node_all.txt'))
        text_node_shell_data_lines = text_node_shell_data.readlines()
        
        shell_node_no=int((len(text_node_shell_data_lines)-1)/(no_beam_elem+1))#no_of_shell_ele_per_section
        
        shell_x_disp=np.zeros((no_beam_elem+1,shell_node_no+1),float)
        shell_x=np.zeros((no_beam_elem+1,shell_node_no+1),float)
        shell_y_disp=np.zeros((no_beam_elem+1,shell_node_no+1),float)
        shell_y=np.zeros((no_beam_elem+1,shell_node_no+1),float)
        shell_z_disp=np.zeros((no_beam_elem+1,shell_node_no+1),float)
        shell_z=np.zeros((no_beam_elem+1,shell_node_no+1),float)
        count_all=1   
        for i_line in range(0,no_beam_elem+1):
            count=0
            while count < shell_node_no: #columns are #node no. ux uy uz X Y Z disp and coordinates 
               shell_z[i_line,count]=float((text_node_shell_data_lines[count_all].split( ))[5])
    
               shell_y[i_line,count]=float((text_node_shell_data_lines[count_all].split( ))[4])
    
               shell_x[i_line,count]=float((text_node_shell_data_lines[count_all].split( ))[6])
    
               shell_z_disp[i_line,count]=float((text_node_shell_data_lines[count_all].split( ))[2])
    
               shell_y_disp[i_line,count]=float((text_node_shell_data_lines[count_all].split( ))[1])
    
               shell_x_disp[i_line,count]=float((text_node_shell_data_lines[count_all].split( ))[3])  
    
            #   print float((text_node_shell_data_lines[count_all].split( ))[5])
               count+=1
               count_all+=1
        shell_z[:,shell_node_no]= shell_z[:,0]   
        shell_y[:,shell_node_no]= shell_y[:,0]  
        shell_x[:,shell_node_no]= shell_x[:,0]  
        shell_z_disp[:,shell_node_no]= shell_z_disp[:,0]  
        shell_y_disp[:,shell_node_no]= shell_y_disp[:,0]
        shell_x_disp[:,shell_node_no]= shell_x_disp[:,0]

        arrow = np.array([[800, 0, 0, 1,0 , 0], [0, -800, 0, 0, -3, 0],[0, 0, 800, 0, 0, 2]])
        XX, YY, ZZ, UU, VV, WW = zip(*arrow)        
        
        scale_local=600
        fig = plt.figure(figsize=(11,8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(scale_local* shell_y_disp+shell_y,scale_local*shell_z_disp+shell_z,(scale_local*shell_x_disp+shell_x), color='c',rstride=1, cstride=3,linewidth=0.1,alpha=1)
          #  surf = ax.plot_wireframe(scale_local*u_X+y,scale_local*w_Z+z,scale_local*v_Y+x, color='k',rstride=2, cstride=2,linewidth=0.1,alpha=1)
        surf = ax.plot_wireframe(shell_y,shell_z,shell_x, rstride=2, cstride=6,color='k', linewidth=0.1,alpha=0.8)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Height')
        ax.auto_scale_xyz([-R*0.1, R*1.2], [-R, R], [-R*0.1, R*1.2])
        ax.quiver(XX, YY, ZZ, UU, VV, WW,length=800)
        xlm=ax.get_xlim3d() #These are two tupples
        ylm=ax.get_ylim3d() #we use them in the next
        zlm=ax.get_zlim3d() #graph to reproduce the magnification from mousing
        ax.set_xlim3d(xlm[0],xlm[1])  
        ax.set_ylim3d(ylm[0],ylm[1])
        ax.set_zlim3d(zlm[0],zlm[1]) 
        ax.view_init(elev=15, azim=-125)
        
        plt.axis('off')
        plt.savefig('out_of_plain_loading_shell.png', format='png', dpi=800)          
         
        
          #  scale_local=10
        fig = plt.figure(figsize=(11,8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(scale_local*u_X+y,scale_local*w_Z+z,scale_local*v_Y+x, color='limegreen',rstride=1, cstride=3,linewidth=0.1,alpha=1)
          #  surf = ax.plot_wireframe(scale_local*u_X+y,scale_local*w_Z+z,scale_local*v_Y+x, color='k',rstride=2, cstride=2,linewidth=0.1,alpha=1)
        surf = ax.plot_wireframe(y,z,x, rstride=2, cstride=6,color='k', linewidth=0.1,alpha=0.8)
        ax.set_xlabel('X axis')
        ax.set_ylabel('Y axis')
        ax.set_zlabel('Height')
        ax.auto_scale_xyz([-R*0.1, R*1.2], [-R, R], [-R*0.1, R*1.2])
        ax.quiver(XX, YY, ZZ, UU, VV, WW,length=800)
        xlm=ax.get_xlim3d() #These are two tupples
        ylm=ax.get_ylim3d() #we use them in the next
        zlm=ax.get_zlim3d() #graph to reproduce the magnification from mousing
        ax.set_xlim3d(xlm[0],xlm[1])  
        ax.set_ylim3d(ylm[0],ylm[1])
        ax.set_zlim3d(zlm[0],zlm[1]) 
        ax.view_init(elev=15, azim=-125)
        
        plt.axis('off')
        plt.savefig('out_of_plain_loading_GBT.png', format='png', dpi=800)  
    ############################################################################################

print("--- %s seconds ---" % (time.time() - start_time))



  

