def greens_generator_v1(data_dict,
                     tmin=1923,
                     tmax=2023,
                     t_steps=100,
                     num_steps=5000,
                     nz=21,
                     tol=1e-2,
                     include_mechanics=False,
                     step_function_input=False,
                     start_at_zero=False):
    keys = ['ALHIC1901_23_5min',
            'ALHIC1903_23_5min',
            'ALHIC1902_23_5min',
            'ALHIC2301_23_5min',
            'ALHIC2201_23_5min',
            'ALHIC1901_23_30min',
            'ALHIC1902_23_30min',
            'ALHIC1902_23_10sec',
            'ALHIC1901_22_30min']
    borehole_geometries = {
        ## top of borehole, bottom of borehole
        'ALHIC1901_22_30min': [162.05, 313.05],
        'ALHIC1901_23_5min': [170.43, 311.04],
        'ALHIC1903_23_5min': [163.82, 311.04],
        'ALHIC1902_23_5min': [104.07, 311.04],
        'ALHIC2301_23_5min': [220.75, 311.04],
        'ALHIC2201_23_5min': [220.0, 311.04],
        'ALHIC1901_23_30min': [170.93, 311.04], 
        'ALHIC1902_23_30min': [104.30, 311.04], 
        'ALHIC1902_23_10sec': [104.30, 311.04]
    }
    ## EStimate of the bottom of the cable
    below_seasonal = {
        ## optical distance below season
        'ALHIC1901_22_30min': [185.05],
        'ALHIC1901_23_5min': [193.43],
        'ALHIC1903_23_5min': [186.82],
        'ALHIC1902_23_5min': [127.07],
        'ALHIC2301_23_5min': [243.75],
        'ALHIC2201_23_5min': [243.0],
        'ALHIC1901_23_30min': [193.93], 
        'ALHIC1902_23_30min': [127.30], 
        'ALHIC1902_23_10sec': [127.30]
    }
    
    year = np.linspace(tmin,tmax, num_steps)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'peru', 'limegreen']
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1,5, figsize=[30,5])
    ax_list = (ax1, ax2, ax3, ax4, ax5)

    counter2 = 0
    counter3 = 0
    counter4 = 0
    counter5 = 0
    
    scale_offset = 0
    off = 0

    adot=-0.01 #Accumulation rate (m/yr)

    
    step_func_len = num_steps / t_steps #please make this a whole number

    if step_function_input == True:
        temp_input = np.zeros((t_steps,num_steps))
        if step_func_len != int(step_func_len):
            raise Exception('num_steps divided by t_steps needs to be an integer!')
    
        for i in range(t_steps):
            start = i*step_func_len
            end = (i*step_func_len) + step_func_len
            temp_input[i,start:end] = 1
            
    if step_function_input == False:
        temp_input = np.eye(num_steps)
        
    depths = {}
    
    fig,ax = plt.subplots()
    ax.imshow(temp_input, aspect='auto')


    for off, (i, c) in enumerate(zip(keys, colors)):
        print(i)
        x = data_dict[i].sel(x=slice(borehole_geometries[i][0],borehole_geometries[i][1])).x - borehole_geometries[i][0]

        y = data_dict[i].sel(x=slice(borehole_geometries[i][0],borehole_geometries[i][1])).tmpw
        
        if i == 'ALHIC1901_23_30min':
            mean = y
            ax1.plot(mean + (off *scale_offset),x, label=i[10:])

            #Model run

            H=x.values[-1] #ice thickness
            depths['ALHIC1901'] = np.linspace(0,H, nz)
            
            #pick the surface temperature by the linear trend
            slope,b = np.polyfit(y.sel(x=slice(below_seasonal[i][0],borehole_geometries[i][1])).x, y.sel(x=slice(below_seasonal[i][0],borehole_geometries[i][1])), 1)
            Ts_all  = np.mean(slope)*y.sel(x=slice(borehole_geometries[i][0],borehole_geometries[i][1])).x + np.mean(b)
            qgeo = np.mean(slope)*2

            
            
            if start_at_zero == False:
                Ts = Ts_all[0].values
                
                print('Geothermal flux (slope) = '+str(qgeo))
                print('intercept = '+str(b))
                print('Ts = ' + str(Ts))
                
                # Instantiate the model class
                m = ice_temperature(
                    Ts=Ts,
                    adot=adot,
                    H=H,
                    qgeo=qgeo,
                    p=1000.,
                    dS=10,
                    nz=nz,
                    tol=tol)
            if start_at_zero == True:
                Ts = 0.
                # Instantiate the model class
                m = ice_temperature(
                    Ts=Ts,
                    adot = 0.,
                    H=H,
                    p=1000.,
                    nz=nz,
                    tol=tol)
            # Set the time step to 5 years and subsample the paleoclimate data to match
            m.ts = year*const.spy
            m.adot_s = np.asarray([adot]*len(year))/const.spy

            if include_mechanics == True:
                # Set velocity terms
                m.Udef=0.
                m.Uslide=0.2/const.spy

                # Longitudinal advection forcing

                m.dTs = 0
                m.dH = 0.3
                m.da = 0.
                m.flags.append('long_advection')

                # Thermal conductivity should be temperature and density dependent; set here
                m.flags.append('temp-dependent')
            m.initial_conditions()


            # Initialize the model to steady state
            
            m.source_terms()
            m.stencil()

            m.numerical_to_steady_state()
            
            if np.isnan(m.T).any():
                raise Exception('NaN introduced during steady state. Check nz and timesteps.')

            ax1.plot(m.T,(m.z-m.H)*-1,c='black',label='Steady State', ls='--')

            AH1901_arr = np.zeros((nz, len(temp_input)))

            for k, surface_temp in enumerate(temp_input):

                # Run the model
                m.Ts_s = surface_temp+Ts
                m.flags.append('save_all')
                m.numerical_transient()
                AH1901_arr[:,k] = m.T
                if np.isnan(m.T).any():
                    raise Exception('NaN introduced during forward model. Check nz and timesteps.')
        elif i[0:9] == 'ALHIC1901' and i != 'ALHIC1901_23_30min':
            mean = np.mean(y, axis=1)
            ci95, ci05 = st.t.interval(alpha=0.95, df=len(data_dict[i])-1, 
                  loc=mean, 
                  scale=st.sem(y, axis=1)) 


            ax1.plot(mean + (off *scale_offset),x, label=i[10:], color=c)
            ax1.fill_betweenx(x,ci05+ (off *scale_offset), ci95+ (off *scale_offset), color='lightgrey')


        elif i[0:9] == 'ALHIC1902':
            mean = np.mean(y, axis=1)
            ci95, ci05 = st.t.interval(alpha=0.95, df=len(data_dict[i])-1, 
                  loc=mean, 
                  scale=st.sem(y, axis=1)) 


            ax2.plot(mean + (off *scale_offset),x, label=i[10:], color=c)
            ax2.fill_betweenx(x,ci05+ (off *scale_offset), ci95+ (off *scale_offset), color='lightgrey')
            if counter2 == 0:
                H=x.values[-1] #ice thickness
                depths['ALHIC1902'] = np.linspace(0,H, nz)
                #pick the surface temperature by the linear trend
                slope,b = np.polyfit(y.sel(x=slice(below_seasonal[i][0],borehole_geometries[i][1])).x, y.sel(x=slice(below_seasonal[i][0],borehole_geometries[i][1])), 1)
                Ts_all  = np.mean(slope)*y.sel(x=slice(borehole_geometries[i][0],borehole_geometries[i][1])).x + np.mean(b)
                qgeo = np.mean(slope)*2


                if start_at_zero == False:
                    
                    Ts = Ts_all[0].values
                    

                    print('Geothermal flux (slope) = '+str(qgeo))
                    print('intercept = '+str(b))
                    print('Ts = ' + str(Ts))
                    # Instantiate the model class
                    m = ice_temperature(
                        Ts=Ts,
                        adot=adot,
                        H=H,
                        qgeo=qgeo,
                        p=1000.,
                        dS=0,
                        nz=nz,
                        tol=tol)

                if start_at_zero == True:
                    
                    Ts = 0.
                     # Instantiate the model class
                    m = ice_temperature(
                        Ts=Ts,
                        adot=adot,
                        H=H,
                        qgeo=qgeo,
                        p=1000.,
                        dS=0,
                        nz=nz,
                        tol=tol)

                # Set the time step to 5 years and subsample the paleoclimate data to match
                m.ts = year*const.spy
                m.adot_s = np.asarray([adot]*len(year))/const.spy
                if include_mechanics == True:
                    
                    # Set velocity terms
                    m.Udef=0.
                    m.Uslide=0.2/const.spy

                    # Longitudinal advection forcing

                    m.dTs = 0
                    m.dH = 0.4
                    m.da = 0.
                    m.flags.append('long_advection')

                    # Thermal conductivity should be temperature and density dependent; set here
                    m.flags.append('temp-dependent')
                m.initial_conditions()


                # Initialize the model to steady state
                m.source_terms()
                m.stencil()

                m.numerical_to_steady_state()
                
                if np.isnan(m.T).any():
                    raise Exception('NaN introduced during steady state. Check nz and timesteps.')

                ax2.plot(m.T,(m.z-m.H)*-1,c='black',label='Steady State', ls='--')
                # Run the model
                AH1902_arr = np.zeros((nz, len(temp_input)))

                for k, surface_temp in enumerate(temp_input):

                    # Run the model

                    m.Ts_s = surface_temp+Ts
                    m.flags.append('save_all')
                    m.numerical_transient()
                    AH1902_arr[:,k] = m.T
                    
                    if np.isnan(m.T).any():
                        raise Exception('NaN introduced during forward model. Check nz and timesteps.')
                counter2 += 1
    #             ax.plot(y,x)
        elif i[0:9] == 'ALHIC1903':

            mean = np.mean(y, axis=1)
            ci95, ci05 = st.t.interval(alpha=0.95, df=len(data_dict[i])-1, 
                  loc=mean, 
                  scale=st.sem(y, axis=1)) 


            ax3.plot(mean + (off *scale_offset),x, label=i[10:], color=c)
            ax3.fill_betweenx(x,ci05+ (off *scale_offset), ci95+ (off *scale_offset), color='lightgrey')
            if counter3 == 0:
                H=x.values[-1] #ice thickness
                depths['ALHIC1903'] = np.linspace(0,H, nz)
                #pick the surface temperature by the linear trend
                slope,b = np.polyfit(y.sel(x=slice(below_seasonal[i][0],borehole_geometries[i][1])).x, 
                                     y.sel(x=slice(below_seasonal[i][0],borehole_geometries[i][1])), 
                                     1)
                Ts_all  = np.mean(slope)*y.sel(x=slice(borehole_geometries[i][0],borehole_geometries[i][1])).x + np.mean(b)
                qgeo = np.mean(slope)*2

                if start_at_zero == False:
                    
                    Ts = Ts_all[0].values
                    

                    print('Geothermal flux (slope) = '+str(qgeo))
                    print('intercept = '+str(b))
                    print('Ts = ' + str(Ts))
                    
                    # Instantiate the model class
                    m = ice_temperature(
                        Ts=Ts,
                        adot=adot,
                        H=H,
                        qgeo=qgeo,
                        p=1000.,
                        dS=15,
                        nz=nz,
                        tol=tol)
                
                if start_at_zero == True:
                    
                    Ts = 0.
                    # Instantiate the model class
                    m = ice_temperature(
                        Ts=Ts,
                        adot=0.,
                        H=H,
                        qgeo=0.,
                        p=1000.,
                        dS=0.,
                        nz=nz,
                        tol=tol)
                # Set the time step to 5 years and subsample the paleoclimate data to match
                m.ts = year*const.spy
                m.adot_s = np.asarray([adot]*len(year))/const.spy

                if include_mechanics == True:

                    # Set velocity terms
                    m.Udef=0.
                    m.Uslide=0.2/const.spy

                    # Longitudinal advection forcing

                    m.dTs = 0
                    m.dH = 0.1
                    m.da = 0.
                    m.flags.append('long_advection')

                    # Thermal conductivity should be temperature and density dependent; set here
                    m.flags.append('temp-dependent')
                m.initial_conditions()


                # Initialize the model to steady state
                m.source_terms()
                m.stencil()
                

                m.numerical_to_steady_state()
                
                if np.isnan(m.T).any():
                    raise Exception('NaN introduced during steady state. Check nz and timesteps.')

                ax3.plot(m.T,(m.z-m.H)*-1,c='black',label='Steady State', ls='--')
                # Run the model
                AH1903_arr = np.zeros((nz, len(temp_input)))

                for k, surface_temp in enumerate(temp_input):

                    # Run the model

                    m.Ts_s = surface_temp+Ts
                    m.flags.append('save_all')
                    m.numerical_transient()
                    AH1903_arr[:,k] = m.T
                    
                    if np.isnan(m.T).any():
                        raise Exception('NaN introduced during forward model. Check nz and timesteps.')
                        
                counter3+=1
                #             ax.plot(y,x)

        elif i[0:9] == 'ALHIC2301':
            mean = np.mean(y, axis=1)
            ci95, ci05 = st.t.interval(alpha=0.95, df=len(data_dict[i])-1, 
                  loc=mean, 
                  scale=st.sem(y, axis=1)) 


            ax4.plot(mean + (off *scale_offset),x, label=i[10:], color=c)
            ax4.fill_betweenx(x,ci05+ (off *scale_offset), ci95+ (off *scale_offset), color='lightgrey')

            if counter4 == 0:
                H=x.values[-1] #ice thickness
                depths['ALHIC2301'] = np.linspace(0,H, nz)
                #pick the surface temperature by the linear trend
                slope,b = np.polyfit(y.sel(x=slice(below_seasonal[i][0],borehole_geometries[i][1])).x, y.sel(x=slice(below_seasonal[i][0],borehole_geometries[i][1])), 1)
                Ts_all  = np.mean(slope)*y.sel(x=slice(borehole_geometries[i][0],borehole_geometries[i][1])).x + np.mean(b)
                qgeo = np.mean(slope)*2

                
                
                if start_at_zero == False:
                    Ts = Ts_all[0].values 
                    
                    print('Geothermal flux (slope) = '+str(qgeo))
                    print('intercept = '+str(b))
                    print('Ts = ' + str(Ts))
                    # Instantiate the model class
                    m = ice_temperature(
                        Ts=Ts,
                        adot=adot,
                        H=H,
                        qgeo=qgeo,
                        p=1000.,
                        dS=0,
                        nz=nz,
                        tol=tol)
                if start_at_zero == True:
                    
                    Ts = 0.
                    # Instantiate the model class
                    m = ice_temperature(
                        Ts=Ts,
                        adot=0.,
                        H=H,
                        qgeo=0.,
                        p=1000.,
                        dS=0.,
                        nz=nz,
                        tol=tol)

                # Set the time step to 5 years and subsample the paleoclimate data to match
                m.ts = year*const.spy
                m.adot_s = np.asarray([adot]*len(year))/const.spy
                
                if include_mechanics == True:

                    # Set velocity terms
                    m.Udef=0.
                    m.Uslide=0.2/const.spy

                    # Longitudinal advection forcing

                    m.dTs = 0
                    m.dH = 0.
                    m.da = 0.
                    m.flags.append('long_advection')

                    # Thermal conductivity should be temperature and density dependent; set here
                    m.flags.append('temp-dependent')
                m.initial_conditions()


                # Initialize the model to steady state
                m.source_terms()
                m.stencil()

                m.numerical_to_steady_state()
                
                if np.isnan(m.T).any():
                    raise Exception('NaN introduced during steady state. Check nz and timesteps.')

                ax4.plot(m.T,(m.z-m.H)*-1,c='black',label='Steady State', ls='--')

                AH2301_arr = np.zeros((nz, len(temp_input)))

                for k, surface_temp in enumerate(temp_input):

                    # Run the model

                    m.Ts_s = surface_temp+Ts
                    m.flags.append('save_all')
                    m.numerical_transient()
                    AH2301_arr[:,k] = m.T
                    
                    if np.isnan(m.T).any():
                        raise Exception('NaN introduced during forward model. Check nz and timesteps.')
                        
                counter4 += 1
    #             ax.plot(y,x)
        elif i[0:9] == 'ALHIC2201':
            mean = np.mean(y, axis=1)
            ci95, ci05 = st.t.interval(alpha=0.95, df=len(data_dict[i])-1, 
                  loc=mean, 
                  scale=st.sem(y, axis=1)) 
            if counter5 == 0:
                H=x.values[-1] #ice thickness
                depths['ALHIC2201'] = np.linspace(0,H, nz)
                #pick the surface temperature by the linear trend
                slope,b = np.polyfit(y.sel(x=slice(below_seasonal[i][0],borehole_geometries[i][1])).x, y.sel(x=slice(below_seasonal[i][0],borehole_geometries[i][1])), 1)
                Ts_all  = np.mean(slope)*y.sel(x=slice(borehole_geometries[i][0],borehole_geometries[i][1])).x + np.mean(b)
                qgeo = np.mean(slope)*2

                ax5.plot(mean + (off *scale_offset),x, label=i[10:], color=c)
                ax5.fill_betweenx(x,ci05+ (off *scale_offset), ci95+ (off *scale_offset), color='lightgrey')
            
                if start_at_zero == False:
                    Ts = Ts_all[0].values
                    

                    print('Geothermal flux (slope) = '+str(qgeo))
                    print('intercept = '+str(b))
                    print('Ts = ' + str(Ts))
                    # Instantiate the model class
                    m = ice_temperature(
                        Ts=Ts,
                        adot=adot,
                        H=H,
                        qgeo=qgeo,
                        p=1000.,
                        dS=0,
                        nz=nz,
                        tol=tol)
                if start_at_zero == True:
                    Ts = 0.
                    # Instantiate the model class
                    m = ice_temperature(
                        Ts=Ts,
                        adot=0.,
                        H=H,
                        qgeo=0.,
                        p=1000.,
                        dS=0.,
                        nz=nz,
                        tol=tol)

                # Set the time step to 5 years and subsample the paleoclimate data to match
                m.ts = year*const.spy
                m.adot_s = np.asarray([adot]*len(year))/const.spy

                if include_mechanics == True:
                    
                    # Set velocity terms
                    m.Udef=0.
                    m.Uslide=0.2/const.spy

                    # Longitudinal advection forcing

                    m.dTs = 0
                    m.dH = 0.4
                    m.da = 0.
                    m.flags.append('long_advection')

                    # Thermal conductivity should be temperature and density dependent; set here
                    m.flags.append('temp-dependent')
                m.initial_conditions()


                # Initialize the model to steady state
                m.source_terms()
                m.stencil()

                m.numerical_to_steady_state()

                if np.isnan(m.T).any():
                    raise Exception('NaN introduced during steady state. Check nz and timesteps.')

                ax5.plot(m.T,(m.z-m.H)*-1,c='black',label='Steady State', ls='--')
                # Run the model
                AH2201_arr = np.zeros((nz, len(temp_input)))

                for k, surface_temp in enumerate(temp_input):

                    # Run the model

                    m.Ts_s = surface_temp+Ts
                    m.flags.append('save_all')
                    m.numerical_transient()
                    AH2201_arr[:,k] = m.T
                    if np.isnan(m.T).any():
                        raise Exception('NaN introduced during forward model. Check nz and timesteps.')
                        
                counter5 += 1
                #             ax.plot(y,x)
                
    borehole_greens = {
        'ALHIC1901': AH1901_arr[::-1],
        'ALHIC1902': AH1902_arr[::-1],
        'ALHIC1903': AH1903_arr[::-1],
        'ALHIC2201': AH2201_arr[::-1],
        'ALHIC2301': AH2301_arr[::-1],
    }
    
    for axis in ax_list:
        axis.invert_yaxis()
    
    return borehole_greens, year, depths
