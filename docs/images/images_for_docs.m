function images_for_docs

addpath ../../../matlab/dynamicalsystems/
addpath ../../../matlab/dmp/
addpath ../../../matlab/functionapproximator/

dim = 1;

figures(1).tau              =    1000;
figures(1).initial_state    =      25;
figures(1).attractor_states =       0;
figures(1).dyn_systems{1}   = ExponentialSystem(dim,figures(1).tau,figures(1).initial_state,figures(1).attractor_states(1,:),4);
figures(1).dyn_systems{2}   = ExponentialSystem(dim,figures(1).tau,figures(1).initial_state,figures(1).attractor_states(1,:),8);
figures(1).analytical       =    1;

figures(2).tau              =       1;
figures(2).initial_state    =       1;
figures(2).attractor_states = [0 0.5];
figures(2).dyn_systems{1}   = ExponentialSystem (dim,figures(2).tau,figures(2).initial_state,figures(2).attractor_states(1,:), 4);
figures(2).dyn_systems{2}   = SpringDamperSystem(dim,figures(2).tau,figures(2).initial_state,figures(2).attractor_states(1,:),20);
figures(2).analytical       =    1;

figures(3).tau              =       1;
figures(3).initial_state    =       1;
figures(3).attractor_states =     0.5;
figures(3).dyn_systems{1}   = Dmp(dim,figures(3).tau,figures(3).initial_state,figures(3).attractor_states(1,:),6,[]); % No goal system
goal_system =   ExponentialSystem(dim,figures(3).tau,figures(3).initial_state,figures(3).attractor_states(1,:),6);
figures(3).dyn_systems{2}   = goal_system;
figures(3).dyn_systems{3}   = Dmp(dim,figures(3).tau,figures(3).initial_state,figures(3).attractor_states(1,:),10,goal_system); % No goal system
figures(3).analytical       =    0;

figures(4).tau              =     0.8;
figures(4).initial_state    =       1;
figures(4).attractor_states =       [0 figures(4).tau];
figures(4).dyn_systems{1}   = ExponentialSystem(dim,figures(4).tau,1,figures(1).attractor_states(1,:),8);
figures(4).dyn_systems{2}   =        TimeSystem(dim,figures(4).tau,0,figures(1).attractor_states(1,:));
figures(4).analytical       =    0;


figures(5).tau              =       1;
figures(5).initial_state    =       1;
figures(5).attractor_states =     0.5;
for dd=1:3
  figures(5).dyn_systems{dd}   = Dmp(dim,figures(3).tau,figures(3).initial_state,figures(3).attractor_states(1,:),6);
  weights = figures(5).dyn_systems{1}.function_approximators(1).weights;
  figures(5).dyn_systems{dd}.function_approximators(1).weights = (dd-1)*30*randn(size(weights));
end
figures(5).analytical       =    0;


n_figures = length(figures);

for ff=1:n_figures
  figure(ff)

  tau = figures(ff).tau;
  dt = tau/250;
  ts = 0:dt:1.5*tau;

  n_colors = length(figures(ff).dyn_systems)*length(figures(ff).attractor_states); % *length(attractor_states);
  colormap(ones(n_colors,3));
  colors = colormap(jet);
  colors_hsv = rgb2hsv(colors);
  colors_hsv(:,2) = 0.3;
  colors_faded = hsv2rgb(colors_hsv); %#ok<NASGU>
  colors_hsv(:,3) = 0.8;
  colors = 0.7*hsv2rgb(colors_hsv); %#ok<NASGU>


  clf
  count=1;
  legend_labels = {};
  clear line_handles
  for aa=1:length(figures(ff).attractor_states)
    attractor_states = figures(ff).attractor_states;
    for dd=1:length(figures(ff).dyn_systems)
      dyn_sys = figures(ff).dyn_systems{dd};
      dyn_sys = dyn_sys.changeAttractorState(attractor_states(aa));
      is_dmp = strcmp(class(dyn_sys),'Dmp');
      
      if (figures(ff).analytical)
        % Analytical solution
        states_analytical = dyn_sys.analyticalSolutionWith(ts);
      end
      
      % Integration
      [dyn_sys first_state] = dyn_sys.reset();
      states_step = zeros(length(ts),length(first_state));
      states_step(1,:) = first_state;
      for tt=2:length(ts)

        %if (strcmp('attractor',cur_test) && (tt==ceil(length(ts)/3)))
        %    % Change the attractor state at t = T/3
        %    attractor_state = attractor_state+0.1;
        %  end

        %  if (strcmp('perturb',cur_test) && (tt==ceil(length(ts)/3)))
        %    % Perturb the state at t = T/3
        %    state((end-dyn_sys.dim+1):end) = 0.2/(dt^sqrt(dyn_sys.order));
        %  end

        %   dyn_sys = dyn_sys.setattractorstate(attractor_state);
        states_step(tt,:) = dyn_sys.integrateStep(dt,states_step(tt-1,:)')';
      end

      axis_handles = [];
      for i_order=1:(dyn_sys.order+1)
        axis_handles(i_order) = subplot(1,3,i_order);
        hold on
      end

      if (figures(ff).analytical)
        line_handles{1,count} = dyn_sys.plotStates(axis_handles,ts,states_analytical);
        set(line_handles{1,count}, 'Color',colors_faded(dd,:),'LineWidth',3);
        legend_labels{end+1} = [ class(dyn_sys) ' (analytical)' ];
      end

      if (is_dmp)
        %dyn_sys.plotStateVectors(states_step);
        line_handles{2,count} = dyn_sys.plotStates(axis_handles,ts,states_step(:,1+(1:dim*3)));
      else
        line_handles{2,count} = dyn_sys.plotStates(axis_handles,ts,states_step);
      end
      set(line_handles{2,count},'LineStyle','--','Color',colors(dd,:)      ,'LineWidth',1);
      legend_labels{end+1} = [ class(dyn_sys) ' (integrated)' ];

      count = count + 1;
    end
  end

  if (ff==1)
    subplot(1,3,1)
    ylabel('temperature (Celcius)','Interpreter','latex');
    subplot(1,3,2)
    ylabel('rate of change of temperature (Celcius/s)','Interpreter','latex');
    %plot2svg('exponential_decay.svg')
  
  elseif (ff==2)
    subplot(1,3,1)
    legend(legend_labels{1:(end/2)})
    %plot2svg('example_systems.svg')
  
  elseif (ff==3)
    for dd=1:size(line_handles,2)
      set(line_handles{2,dd},'LineStyle','-','Color',0.8*colors_faded(dd,:)      ,'LineWidth',2);
    end
    set(line_handles{2,2},'LineStyle','--');
    subplot(1,3,2)
    legend('DMP without delayed goal system','delayed goal system','DMP with delayed goal system','Location','South')
    %plot2svg('dmp_and_goal_system.svg')

  elseif (ff==4)
    for dd=1:size(line_handles,2)
      set(line_handles{2,dd},'LineStyle','-','Color',0.8*colors_faded(dd,:)      ,'LineWidth',2);
    end
    set(line_handles{2,1},'LineStyle','-');
    set(line_handles{2,4},'LineStyle','-');
    delete(line_handles{2,2})
    delete(line_handles{2,3})
    subplot(1,3,1)
    legend('Exponential phase system','Constant velocity phase system','Location','North')
    %plot2svg('phase_systems.svg')
  elseif (ff==5)
    for dd=1:size(line_handles,2)
      set(line_handles{2,dd},'LineStyle','-','Color',0.8*colors_faded(dd,:)      ,'LineWidth',2);
    end
    %set(line_handles{2,1},'LineStyle','-');
    %set(line_handles{2,4},'LineStyle','-');
    subplot(1,3,1)
    legend('DMP (no forcing term)','DMP (forcing term 1)','DMP (forcing term 2)','Location','North')
    %plot2svg('dmp_forcing_terms.svg')
  else
  end
end

disp('Call "demodmp" to get the final plot with all the systems.')

end