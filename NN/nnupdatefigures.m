function nnupdatefigures(nn, fhandle, opts, i)
%NNUPDATEFIGURES updates figures during training
if isfield(opts,'plot') && opts.plot > 0 && i > 1
    x_ax = 1:i;
    % create legend
    if opts.validation == 1
        M = {'Training','Validation'};
    else
        M = {'Training'};
    end
    
    % create data for plots
    plot_x = x_ax';
    plot_ye = nn.eval.train.error';
    plot_ya = nn.eval.train.accuracy';
    
    % add error and classification accuracy on validation data if present
    if opts.validation == 1
        plot_x = [plot_x, x_ax'];
        plot_ye = [plot_ye, nn.eval.val.error'];
        plot_ya = [plot_ya, nn.eval.val.accuracy'];
    end
    
    % plotting
    figure(fhandle);
    if ~isfield(opts, 'netname')
        opts.netname = 'default';
    end
    if opts.plot == 3
        p1 = subplot(1,2,1);
        plot(plot_x, plot_ye);
        xlabel('Number of epochs'); ylabel('Error');
        title(['Error (' opts.netname ': ' mat2str(nn.size) ' ' nn.activation_function '-' nn.output ')']);
        legend(p1, M,'Location','NorthEast');
        set(p1, 'Xlim',[0,opts.numepochs + 1]);
        grid on;
        
        p2 = subplot(1,2,2);
        plot(plot_x, plot_ya);
        xlabel('Number of epochs'); ylabel('Accuracy');
        title(['Accuracy (' opts.netname ': ' mat2str(nn.size) ' ' nn.activation_function '-' nn.output ')']);
        legend(p2, M,'Location','SouthEast');
        set(p2, 'Xlim',[0,opts.numepochs + 1]);
        grid on;
        
    elseif opts.plot == 1
        p = plot(plot_x, plot_ye);
        xlabel('Number of epochs'); ylabel('Error');
        title(['Error (' opts.netname ': ' mat2str(nn.size) ' ' nn.activation_function '-' nn.output ')']);
        legend(p, M,'Location','NorthEast');
        set(gca, 'Xlim',[0,opts.numepochs + 1]);
        grid on;
        
    elseif opts.plot == 2
        p = plot(plot_x, plot_ya);
        xlabel('Number of epochs'); ylabel('Accuracy');
        title(['Accuracy (' opts.netname ': ' mat2str(nn.size) ' ' nn.activation_function '-' nn.output ')']);
        legend(p, M,'Location','SouthEast');
        set(gca, 'Xlim',[0,opts.numepochs + 1]);
        grid on;
        
    end
    drawnow;
end
end
