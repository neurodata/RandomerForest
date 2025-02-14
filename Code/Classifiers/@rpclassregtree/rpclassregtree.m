classdef rpclassregtree

    properties
        method = 'regression';
          node = zeros(0,1);
        parent = zeros(0,1);
         class = zeros(0,1);
           var = zeros(0,1);
           cut = cell(0,1);
      children = zeros(0,2);
      nodeprob = zeros(0,1);
       nodeerr = zeros(0,1);
      nodesize = zeros(0,1);
         npred = 0;
       catcols = [];
         prior = [];
      nclasses = 1;
          cost = [];
     classprob = [];
      impurity = [];
    classcount = [];
     classname = {};
     prunelist = zeros(0,1);
         alpha = [];
    ntermnodes = [];
         names = {};
prunecriterion = '';
     minparent = [];
       minleaf = [];
  nvartosample = [];
   mergeleaves = [];
splitcriterion = '';
       qetoler = [];
      varassoc = {};
        varimp = [];
     nvarsplit = [];
       surrvar = {};
       surrcut = {};
      surrflip = {};
      catsplit = []; % for backwards compatibility with version prior to 9a
           rpm = [];
       isdelta = [];
    end
    
    methods
        function a = rpclassregtree(X,Y,varargin)
            if nargin==1 && isa(X,'struct')
                a = struct2tree(a,X);            % convert from struct to tree
            else
                narginchk(2,Inf);
                a = rptreefit(a,X,Y,varargin{:});  % calls local version of rptreefit
                a.prunelist = zeros(0,1);
            end
        end % rpclassregtree constructor
        
        function Yhat = rptreepredict(Tree,X)
            Yhat = cell(size(X,1),1);
            noderows = cell(0,length(Tree.node));
            noderows{1} = 1:size(X,1);
            internalnodes = Tree.node(Tree.var ~= 0);
            internalnodes = internalnodes';
            leafnodes = Tree.node(Tree.var == 0);
            leafnodes = leafnodes';
            for node = internalnodes
                cut = Tree.cut{node};
                projection = Tree.rpm(:,node);
                Xpro = X(noderows{node},:)*projection;
                ch = Tree.children(node,:);
                noderows{ch(1)} = noderows{node}(Xpro < cut);
                noderows{ch(2)} = noderows{node}(Xpro >= cut);
            end
            for node = leafnodes
                Yhat(noderows{node}) = Tree.classname(Tree.class(node));
            end
        end     %function rptreepredict
        
        function classprob = rpclassprob(Tree,X,varargin)
            % feature for which importance is desired to be computed along
            % with permuted values for that feature can be passed in as
            % optional third and fourth arguments
            if nargin == 4
                projection_imp = varargin{1};   % importance feature
                X_imp = varargin{2};    % permuted values for this feature
            end
            classprob = NaN(size(X,1),length(Tree.classname));
            noderows = cell(0,length(Tree.node));
            noderows{1} = 1:size(X,1);
            internalnodes = Tree.node(Tree.var ~= 0);
            internalnodes = internalnodes';
            leafnodes = Tree.node(Tree.var == 0);
            leafnodes = leafnodes';
            if nargin == 4
                for node = internalnodes
                    cut = Tree.cut{node};
                    projection = Tree.rpm(:,node);
                    if isequal(projection,projection_imp)
                        Xpro = X_imp(noderows{node});
                    else
                        Xpro = X(noderows{node},:)*projection;
                    end
                    ch = Tree.children(node,:);
                    noderows{ch(1)} = noderows{node}(Xpro < cut);
                    noderows{ch(2)} = noderows{node}(Xpro >= cut);
                end
            else
                for node = internalnodes
                    cut = Tree.cut{node};
                    projection = Tree.rpm(:,node);
                    Xpro = X(noderows{node},:)*projection;
                    ch = Tree.children(node,:);
                    noderows{ch(1)} = noderows{node}(Xpro < cut);
                    noderows{ch(2)} = noderows{node}(Xpro >= cut);
                end
            end
            for node = leafnodes
                classprob(noderows{node},:) = repmat(Tree.classprob(node,:),length(noderows{node}),1);            
            end
        end     %function rpclassprob
        
        function Depth = tree_depth(Tree)
            Depth = 1;
            CurrentNode = Tree.numnodes;
            while Tree.parent(CurrentNode) ~= 0
                CurrentNode = Tree.parent(CurrentNode);
                Depth = Depth + 1;
            end
        end % function tree_depth
    end     %methods block
    
        methods(Hidden = true)
        function b = fieldnames(a)
            b = properties(a);
        end
        
        % Methods that we inherit, but do not want
        function a = fields(varargin),     throwUndefinedError(); end
        function a = ctranspose(varargin), throwUndefinedError(); end
        function a = transpose(varargin),  throwUndefinedError(); end
        function a = permute(varargin),    throwUndefinedError(); end
        function a = reshape(varargin),    throwUndefinedError(); end
        function a = cat(varargin),        throwNoCatError(); end
        function a = horzcat(varargin),    throwNoCatError(); end
        function a = vertcat(varargin),    throwNoCatError(); end
    end
    
    methods(Hidden = true, Static = true)
        function a = empty(varargin)
            error(message('stats:classregtree:NoEmptyAllowed', upper( mfilename )));
        end
    end
    
    methods
        % Get cuts. This method is here to address the backwards
        % incompatibility introduced in 9a. Prior to 9a classregtree had a
        % numeric array 'cut' and a cell array 'catsplit' for categorical
        % cuts. As of 9a, classregtree has a cell array 'cut' that combines
        % both.
        function newcut = get.cut(tree)
            if iscell(tree.cut)
                newcut = tree.cut;
            else
                % Convert old cuts into new cuts
                csplit = tree.catsplit;
                nnode = length(tree.node);
                newcut = cell(nnode,1);
                for inode=1:nnode
                    if tree.var(inode)<0 % categorical variable
                        icat = tree.cut(inode);
                        newcut{inode} = csplit(icat,:);
                    else % continuous variable (>0) or leaf (=0)
                        newcut{inode} = tree.cut(inode);
                    end
                end
            end
        end
    end
    
    methods(Static=true,Hidden=true)
        function [X,Y,W,classnames,Yorig] = preparedata(X,Y,W,doclass)
            %PREPAREDATA Perform initial data checks and transformations.
            %
            % [XOUT,YOUT,WOUT] = PREPAREDATA(X,Y,W,DOCLASS) processes matrix of
            % predictor values X and vector of true response Y with weights W. For
            % regression, Y is a numeric vector of regressed values. For
            % classification, Y is a categorical vector of true class labels
            % represented as an array of either char's or integers or a cell array of
            % strings or a categorical array. X is a numeric matrix with one row per
            % observation and one column per input variable. W is a numeric
            % column-vector with one element per observation. The number of rows in X
            % must match the number of elements in Y. DOCLASS is a logical flag, true
            % for classification and false for regression.
            %
            % PREPAREDATA removes rows with nothing but missing values (NaN's) in X,
            % rows without a valid class label in Y (for classification), and rows with
            % NaN's in Y (for regression). For regression, it simply returns X, Y and W
            % with these rows removed as XOUT, YOUT and WOUT.
            %
            % For classification, [XOUT,YOUT,WOUT] = PREPAREDATA(X,Y,W,true) returns a
            % numeric vector YOUT of group indices for groups found in input labels Y.
            %
            % [XOUT,YOUT,WOUT,CLASSNAMES,YORIG] = PREPAREDATA(X,Y,W,true) also returns
            % class names and YORIG, categorical Y data for classification. The class
            % names are for the groups found in input labels Y. YORIG is always a cell
            % array of strings with the original Y labels while YOUT is always a vector
            % of numeric indices irrespective of the type of Y.
            %
            % See also GRP2IDX.
        
            % Check input type
            if ~isfloat(X)
                error(message('stats:classregtree:preparedata:BadXType'));
            end
            
            % Check input size
            if isempty(X) || isempty(Y)
                error(message('stats:classregtree:preparedata:EmptyXorY'));
            end
            if size(X,1)~=length(Y)
                error(message('stats:classregtree:preparedata:InputSizeMismatch'));
            end
 
            % Y must be numeric for regression
            if ~doclass && ~isfloat(Y)
                error(message('stats:classregtree:preparedata:BadYType'));
            end
            if ~ischar(Y)
                Y = Y(:);
            end
            
            % Check weights
            if ~isfloat(W) || length(W)~=size(X,1) || ~isvector(W)
                error(message('stats:classregtree:preparedata:BadW'));
            end
            if any(W<0) || all(W==0)
                error(message('stats:classregtree:preparedata:NegativeWeights'));
            end
            W = W(:);
            
            % Get rid of NaN's in the vector of responses
            t = [];
            if ~doclass
                t = isnan(Y);
            end
            if any(t)
                X(t,:) = [];
                Y(t) = [];
                W(t) = [];
            end
            
            % Make class labels
            classnames = {};
            Yorig = Y;
            if doclass
                if isa(Y,'categorical')
                    Y = removecats(Y);
                end
                [Y,classnames] = grp2idx(Y);
                t = isnan(Y);
                if any(t)
                    X(t,:) = [];
                    Y(t) = [];
                    Yorig(t,:) = [];
                    W(t) = [];
                end
                if isempty(X)
                    error(message('stats:classregtree:preparedata:EmptyXorYafterNaNGroup'));
                end
            end
            
            % Get rid of instances that have NaN's in all vars
            t = all(isnan(X),2);
            if any(t)
                X(t,:) = [];
                Y(t) = [];
                Yorig(t,:) = [];
                W(t) = [];
            end
            if isempty(X)
                error(message('stats:classregtree:preparedata:EmptyXorYafterNaN'));
            end
            
            % Get rid of observation with zero weights
            t = W==0;
            if any(t)
                X(t,:) = [];
                Y(t) = [];
                Yorig(t,:) = [];
                W(t) = [];
            end
            if isempty(X)
                error(message('stats:classregtree:preparedata:EmptyXorYafterZeroW'));
            end
        end
        
        function varnames = preparevars(varnames,nvars)
            %PREPAREVARS Perform initial variable check and transformations.
            %
            % VARNAMES = PREPAREVARS(VARNAMES,NVARS) checks the size of the input
            % vector of variable names VARNAMES against the expected size NVARS,
            % converts VARNAMES to a cell array if VARNAMES are char's and assigns
            % default variable names if none are supplied.
            if ~isempty(varnames)
                if ischar(varnames)
                    varnames = cellstr(varnames);
                end
                if ~iscellstr(varnames) || numel(varnames)~=nvars
                    error(message('stats:classregtree:preparevars:BadNames', nvars));
                end
            else
                varnames = nvars;
            end
        end
        
        function [prior,cost,removerows] = priorandcost(prior,cost,cnames,wj,id)
            %PRIORANDCOST Perform initial check of prior probabilities and cost matrix.
            %
            % [PRIOR,COST,REMOVEROWS] = PRIORANDCOST(PRIOR,COST,CNAMES,WJ,ID) checks
            % the input vector of prior probabilities and the input cost matrix. It
            % returns default prior and cost arrays if empty prior and cost were
            % supplied. It also returns indices of rows to be removed in case the prior
            % probability or misclassification cost for this class is zero.
            
            % Convert to row-vector.
            prior = prior(:)';
            
            % Get number of classes
            nclass = length(cnames);
            if ~iscellstr(cnames) || nclass==0
                error(message('stats:classregtree:priorandcost:BadClassNames'));
            end
            
            % Check class id's
            if isempty(id) || ~isnumeric(id) || max(id)>nclass || min(id)<1
                error(message('stats:classregtree:priorandcost:BadClassIndex', nclass));
            end
            
            % Check cumulative weights for classes
            if ~isnumeric(wj) || length(wj)~=nclass
                error(message('stats:classregtree:priorandcost:BadClassWeights', nclass));
            end
            
            % Check prior
            if isempty(prior) || strcmpi(prior,'empirical')
                prior = wj;
            elseif strcmpi(prior,'equal')
                prior = ones(1,nclass);
            elseif isstruct(prior)
                if ~isfield(prior,'group') || ~isfield(prior,'prob')
                    error(message('stats:classregtree:priorandcost:PriorStructMissingField'));
                end
                idx = getclassindex(cnames,prior.group);
                if any(idx==0)
                    j = find(idx==0);
                    error(message('stats:classregtree:priorandcost:PriorStructMissingGroup', cnames{ j( 1 ) }));
                end
                prior = prior.prob(idx);
            end
            if ~isfloat(prior) || length(prior)~=nclass || any(prior<0) || all(prior==0)
                error(message('stats:classregtree:priorandcost:BadPrior', nclass));
            end

            % Find observations for classes with zero prior probabilities
            prior(wj==0) = 0;
            removerows = false(1,length(id)); % remove no rows
            zeroprior = prior==0;
            if any(zeroprior)
                removerows = zeroprior(id);
            end
            if all(prior==0)
                error(message('stats:classregtree:priorandcost:ZeroPriorForObservedClasses'));
            end
            
            % Normalize priors in such a way that the priors in present
            % classes add up to one.
            prior = prior/sum(prior);

            % Get costs
            if isempty(cost)
                cost = ones(nclass) - eye(nclass);
            else
                if isstruct(cost)
                    if ~isfield(cost,'group') || ~isfield(cost,'cost')
                        error(message('stats:classregtree:priorandcost:CostStructMissingField'));
                    end
                    idx = getclassindex(cnames,cost.group);
                    if any(idx==0)
                        j = find(idx==0);
                        error(message('stats:classregtree:priorandcost:CostStructMissingGroup', cnames{ j( 1 ) }));
                    end
                    cost = cost.cost(idx,idx);
                end
                if ~isequal(size(cost),nclass*ones(1,2))
                    error(message('stats:classregtree:priorandcost:CostSizeMismatch', nclass, nclass));
                elseif any(diag(cost)~=0)
                    error(message('stats:classregtree:priorandcost:CostNonzeroDiag'));
                elseif any(cost<0)
                    error(message('stats:classregtree:priorandcost:CostNegative'));
                elseif ~any(any(cost>0)) && nclass>1
                    error(message('stats:classregtree:priorandcost:CostWithoutPositiveValues'));
                end
            end
        
            % Find observations for classes with zero misclassification costs
            if nclass>1
                zerocost = all(cost==0,2)';
                if any(zerocost)
                    removerows = removerows | zerocost(id);
                end
                if sum(removerows)==length(id)
                    error(message('stats:classregtree:priorandcost:ZeroCostForObservedClasses'));
                end
            end
        end
    end
end % classdef

function throwNoCatError()
error(message('stats:classregtree:NoCatAllowed', upper( mfilename )));
end

function throwUndefinedError()
st = dbstack;
name = regexp(st(2).name,'\.','split');
error(message('stats:classregtree:UndefinedFunction', name{ 2 }, mfilename));
end

function Tree=rptreefit(Tree,X,Y,varargin)

% Process inputs

if isnumeric(Y)
   Method = 'regression';
else
   Method = 'classification';
end

okargs =   {'priorprob'   'cost'  'splitcriterion' ...
            'splitmin' 'minparent' 'minleaf' ...
            'nvartosample' 'mergeleaves' 'categorical' 'prune' 'method' ...
            'qetoler' 'names' 'weights' 'surrogate' 'skipchecks' ...
            'stream' 'rho'    'mdiff' 'RandomMatrix'  'nnzs'  'p' 'dprime',...
            'DownsampleNode'    'MaxNodeSize'   'ImHeight'  'ImWidth'};
defaults = {[]            []      'gdi'                        ...
            []         2          1                          ...
            'all'          'on'          []            'off'    Method      ...
            1e-6      {}       []        'off'      false ...
            []  1/size(X,2)   'off'   'sparse' 2   [] []  false   100   []  []};

[Prior,Cost,Criterion,splitmin,minparent,minleaf,...
    nvartosample,Merge,categ,Prune,Method,qetoler,names,W,surrogate,...
    skipchecks,Stream,rho,mdiff,RandomMatrix,nnzs,p,dprime,DownsampleNode,...
    MaxNodeSize,ImHeight,ImWidth,~,extra] = internal.stats.parseArgs(okargs,defaults,varargin{:});

% For backwards compatibility. 'catidx' is a synonym for 'categorical'
for j=1:2:length(extra)
    if strncmp(extra{j},'catidx',length(extra{j}))
        categ = extra{j+1};
    else
        error(message('stats:classregtree:BadParamNameAndNotCatidx', extra{ j }));
    end
end

% Decode method
if ~ischar(Method) || isempty(Method) || ~(Method(1)=='c' || Method(1)=='r')
   error(message('stats:classregtree:BadMethod'));
elseif Method(1)=='c'
   Method = 'classification';
else
   Method = 'regression';
end

% Classification or regression?
doclass = strcmpi(Method(1),'c');

% Preprocess data
if isempty(W)
    W = ones(size(X,1),1);
end

%sort Ys and corresponding Xs so that cnames is ordered consistently across
%all trees in a forest
[Y,sortidx] = sort(Y);
X = X(sortidx,:);

if skipchecks
    if doclass
        % If checks are skipped, Y must be of type ClassLabel.
        [Y,cnames] = grp2idx(Y);
    end
else
    [X,Y,W,cnames] = classregtree.preparedata(X,Y,W,doclass);
end

[N,nvars] = size(X);
%if strcmp(mdiff,'on')
%    nvars = nvars + 1;
%end

% Process variable names
if ~skipchecks
    %if strcmp(mdiff,'on')
    %    names = classregtree.preparevars(names,nvars-1);
    %else
        names = classregtree.preparevars(names,nvars);
    %end
end

% Fill out criterion, class labels and matrix for classification
if doclass
   switch(Criterion)
    %                Criterion function   Is it an impurity measure?
    %                ------------------   --------------------------
    case 'gdi',      critfun = @gdi;      isimpurity = true;
    case 'twoing',   critfun = @twoing;   isimpurity = false;
    case 'deviance', critfun = @deviance; isimpurity = true;
    otherwise,     error(message('stats:classregtree:BadSplitCriterion'))
   end
   
   % Get binary matrix, C(i,j)==1 means point i is in class j
   nClasses = length(cnames);
   C = false(N,nClasses);
   C(sub2ind([N nClasses],(1:N)',Y)) = 1;   
   WC = bsxfun(@times,C,W);
   Wj = sum(WC,1);
else
   C = Y(:);
   isimpurity = [];
   critfun = [];
end

if isempty(p)
    p = 1/nClasses;
end

% Process prior and cost
if doclass
    % Get prior and cost
    if skipchecks
        if ~isstruct(Prior) || ~isstruct(Cost)
            error(message('stats:classregtree:BadPriorOrCostForSkipchecks'));
        end
        idx = getclassindex(cnames,Prior.group);
        Prior = Prior.prob(idx);
        idx = getclassindex(cnames,Cost.group);
        Cost = Cost.cost(idx,idx);
    else
        [Prior,Cost,removerows] = classregtree.priorandcost(Prior,Cost,cnames,Wj,Y);
        if any(removerows)
            X(removerows,:) = [];
            C(removerows,:) = [];
            WC(removerows,:) = [];
            Wj = sum(WC,1);
        end
        idx = Wj>0;
        W = sum(bsxfun(@times,WC(:,idx),Prior(idx)./Wj(idx)),2);
    end
    
    % Adjust prior to take costs into account
    % pratio is a multiplicative factor for class probabilities
    Cj = sum(Cost,2)';
    pratio = nClasses*Cj / sum(Cj);
    if ~isa(pratio,'double')
        pratio = double(pratio);
    end
else % regression
   pratio = 1;
end

% Clear WC to release memory
WC = [];

% Check and adjust node sizes.
if ~isempty(splitmin) && ~isempty(minparent)
    error(message('stats:classregtree:BothSplitminAndMinparentNotAllowed'));
end
if ~isempty(splitmin)
    if ~isnumeric(splitmin) || ~isscalar(splitmin)
        error(message('stats:classregtree:BadSplitmin'));
    end
    minparent = splitmin;
end
if ~isempty(minparent) && (~isnumeric(minparent) || ~isscalar(minparent))
    error(message('stats:classregtree:BadMinparent'));
end
if isempty(minparent)
    minparent = 2;
end
if ~isempty(minleaf) && (~isnumeric(minleaf) || ~isscalar(minleaf))
    error(message('stats:classregtree:BadMinleaf'));
end
if minleaf<1
    error(message('stats:classregtree:MinleafLessThanOne'));
end
if minparent<1
    error(message('stats:classregtree:MinparentLessThanOne'));
end
minparent = max(minparent,2*minleaf);

% Compute surrogate splits?
if ~strcmpi(surrogate,'on') && ~strcmpi(surrogate,'off')
    error(message('stats:classregtree:BadSurrogate'));
end
surrogate = strcmpi(surrogate,'on');

% Set the number of vars to be selected at random for splits
% Get number of features to sample
success = false;
if strcmpi(nvartosample,'all')
    %if strcmp(mdiff,'on')
    %    nvartosample = nvars-1;
    %else
        nvartosample = nvars;
    %end
    success = true;
end
if isnumeric(nvartosample) && nvartosample>0
    %if strcmp(mdiff,'on')
    %    nvartosample = ceil(nvartosample) - 1;
    %else
        nvartosample = ceil(nvartosample);
    %end
    success = true;
end
if ~success
    error(message('stats:classregtree:BadNumberOfRandomFeatures', nvars));
end
nusevars = nvartosample;

% Check prune parameter
if ~strcmpi(Prune,'on') && ~strcmpi(Prune,'off')
    error(message('stats:classregtree:BadPrune'));
end

% Check merge parameter
if ~strcmpi(Merge,'on') && ~strcmpi(Merge,'off')
    error(message('stats:classregtree:BadMergeLeaves'));
end

% Tree structure fields ([C] only for classification trees):
%  .method     method
%  .node       node number
%  .parent     parent node number
%  .class      class assignment for points in this node if treated as a leaf
%  .var        column j of X matrix to be split, or 0 for a leaf node,
%              or -j to treat column j as categorical
%  .cut        cutoff value for split (Xj<cutoff goes to left child node),
%              or a cell array of left and right values if var is negative
%  .children   matrix of child nodes (2 cols, 1st is left child)
%  .nodeprob   probability p(t) for this node
%  .nodeerr    resubstitution error estimate r(t) for this node
%  .nodesize   number of points at this node
%  .prunelist  list of indices that define pruned subtrees.  One entry per
%              node.  If prunelist(j)=k then, at the kth level of pruning,
%              the jth node becomes a leaf (or drops off the tree if its
%              parent also gets pruned).
%  .alpha      vector of complexity parameters for each pruning cut
%  .ntermnodes vector of terminal node counts for each pruning cut
%  .classprob  [C] vector of class probabilities
%  .classname  [C] names of each class
%  .classcount [C] count of members of each class
%  .nclasses   [C] number of classes
%  .cost       [C] misclassification cost

N = size(X,1);
Wtot = sum(W);
M = 2*ceil(N/minleaf)-1;% number of tree nodes for space reservation
nodenumber = zeros(M,1);
parent = zeros(M,1);
yfitnode = zeros(M,1);
cutvar = zeros(M,1);
cutpoint = cell(M,1);
children = zeros(M,2);
nodeprob = zeros(M,1);
resuberr = zeros(M,1);
nodesize = zeros(M,1);
rpm = sparse(nvars,M);    %initialize cell array for storing proj matrices
isdelta = false(M,1);
if doclass
   classprob = zeros(M,nClasses);
   classcount = zeros(M,nClasses);
   if isimpurity
       impurity = zeros(M,1);
   end
   ybar = [];
end

iscat = false(nvars,1); iscat(categ) = 1;
nvarsplit = zeros(1,nvars);
varimp = [];
varassoc = [];
surrvar = {};
surrcut = {};
surrflip = {};
if surrogate
    varimp = zeros(1,nvars);
    varassoc = repmat({[]},M,1); % var associations
    surrvar = repmat({[]},M,1); % list of surrogate split vars for each node
    surrcut = repmat({{}},M,1); % list of surrogate split cuts for each node
    surrflip = repmat({[]},M,1);% list of flips: +1 do not flip, -1 flip
end

nodenumber(1) = 1;

assignednode = cell(M,1);% list of instances assigned to this node
assignednode{1} = 1:N;
nextunusednode = 2;

% Keep processing nodes until done
tnode = 1;
Labels = unique(Y);
K = length(Labels);
pairs = zeros(K-1,2);
npairs = K-1;
pairs(:,1) = 1:npairs;
pairs(:,2) = pairs(:,1) + 1;
mu_diff = zeros(nvars,npairs);
if isempty(p)
    %p = hygepdf(1,size(X,2),1,nvartosample);
    p = 1/npairs;
end

%compute class conditional difference in means and scale each dimension by
%the average of the class-conditional standard deviations for that
%dimension
if strcmp(mdiff,'all') && K > 1
    for i = 1:npairs
        mu_diff(:,i) = transpose(mean(X(Y==Labels(pairs(i,2)),:)) - mean(X(Y==Labels(pairs(i,1)),:)));
        %mu_diff(:,i) = mu_diff(:,i)./transpose(mean(cat(1,std(X(Y==Labels(pairs(i,1)),:)),std(X(Y==Labels(pairs(i,2)),:)))));
    end
end

while(tnode < nextunusednode)
   % Record information about this node
   noderows = assignednode{tnode};
   Nt = length(noderows);
   Cnode = C(noderows,:);
   Wnode = W(noderows);
   Wt = sum(Wnode);
   
   % Do we want to downsample the node observations?   
   if DownsampleNode && NodeSize > MaxNodeSize
      NodeSample = randperm(NodeSize,MaxNodeSize);
      Csub = Cnode(NodeSample,:);
      Wsub = Wnode(NodeSample,:);
   else
      NodeSample = 1:length(noderows);
      Csub = Cnode;
      Wsub = Wnode;
   end
   
   if doclass
      % Compute class probabilities and related statistics for this node
      Njt = sum(Cnode,1);    % number in class j at node t
      Pjandt = sum(bsxfun(@times,Cnode,Wnode),1);
      Pjgivent = Pjandt / sum(Pjandt);
      misclasscost = Pjgivent * Cost;
      [mincost,nodeclass] = min(misclasscost);
      yfitnode(tnode) = nodeclass;
      nodeprob(tnode) = Wt;
      classprob(tnode,:) = Pjgivent;
      classcount(tnode,:) = Njt;
      impure = sum(Pjgivent>0)>1;
      if isimpurity
          Pcorr = Pjgivent.*pratio;
          impurity(tnode) = feval(critfun,Pcorr/sum(Pcorr));
      end
   else
      % Compute variance and related statistics for this node
      ybar = sum(Cnode.*Wnode)/Wt;
      yfitnode(tnode) = ybar;
      nodeprob(tnode) = Wt/Wtot;
      sst = sum((Cnode-ybar).^2.*Wnode);% total sum of squares at this node
      mincost = sst / Wt;
      impure = (mincost > qetoler*resuberr(1));
   end
   bestcrit          = -Inf;
   nodesize(tnode)   = Nt;
   resuberr(tnode)   = mincost;
   cutvar(tnode)     = 0;
   cutpoint{tnode}   = 0;
   children(tnode,:) = 0;
   if surrogate
       varassoc(tnode) = {[]};
       surrvar(tnode) = {[]};
       surrcut(tnode) = {{}};
       surrflip(tnode) = {[]};
   end
   
   % Consider splitting this node
    if (Nt>=minparent) && impure      % split only large impure nodes
        Xnode = X(noderows,:);
        Ynode = Y(noderows);
        if strcmp(mdiff,'node') && K > 1
            for i = 1:npairs
                mu_diff(:,i) = transpose(mean(Xnode(Ynode==Labels(pairs(i,2)),:)) - mean(Xnode(Ynode==Labels(pairs(i,1)),:)));
                %mu_diff(:,i) = mu_diff(:,i)./transpose(mean(cat(1,std(X(Y==Labels(pairs(i,1)),:)),std(X(Y==Labels(pairs(i,2)),:)))));
            end
        end
        
        if strcmp(RandomMatrix,'image')
            promat = structured_rp(ImHeight,ImWidth,[],[],nusevars,[],false);
        elseif strcmp(RandomMatrix,'image-control')
            promat = structured_rp(ImHeight,ImWidth,[],[],nusevars,[],true);
        else        
            if (strcmp(mdiff,'all') || strcmp(mdiff,'node')) && K > 1
                promat = srpmat(nvars,nusevars,RandomMatrix,rho,nnzs,dprime);    %random projection matrix
                md_ind = rand(size(mu_diff,2),1) <= p;
                promat = [mu_diff(:,md_ind) promat];
                md_idx = 1:sum(md_ind);   %Indices of where the mean difference vectors are in the matrix
                iscat2 = cat(1,false(sum(md_ind),1),iscat);
                %nvarsplit2 = cat(2,zeros(1,sum(md_ind)),nvarsplit);
                nvarsplit = cat(2,zeros(1,sum(md_ind)),nvarsplit);
            else
                promat = srpmat(nvars,nusevars,RandomMatrix,rho,nnzs,dprime);    %random projection matrix
                iscat2 = iscat;
                %nvarsplit2 = nvarsplit;
            end
        end
        
        Xnode = Xnode*promat; %project Xnode onto random bases of promat
        Xsub = Xnode(NodeSample,:);
      
        bestvar = 0;
        bestcut = 0;
      
      % Find the best of all possible splits
%       for jvar=1:size(Xnode,2)
        for jvar = 1:size(Xsub,2)

            % Categorical variable?
            %xcat = iscat2(jvar);
            xcat = false;

            % Get rid of missing values and sort this variable
            %          idxnan = isnan(Xnode(:,jvar));
            idxnan = isnan(Xsub(:,jvar));
            idxnotnan = find(~idxnan);
            if isempty(idxnotnan)
             continue;
            end

            %          [x,idxsort] = sort(Xnode(idxnotnan,jvar));
            [x,idxsort] = sort(Xsub(idxnotnan,jvar));
            idx = idxnotnan(idxsort);
            %          c = Cnode(idx,:);
            %          w = Wnode(idx);
            c = Csub(idx,:);
            w = Wsub(idx,:);

            % Downweight the impurity (for classification) or node mse (for
            % regression) by the fraction of observations that are being
            % split. Twoing already penalizes splits with low pL and pR.
            crit0U = 0;
            crit0  = 0;
            if doclass 
                if isimpurity % twoing crit does not need to be offset
                 % crit0U = P(t0-tU)*i(t0)
                 % crit0  = P(t0)*i(t0)
                %                  Pmis = sum(Wnode(idxnan));
                 Pmis = sum(Wsub(idxnan));
                 crit0U = impurity(tnode)*(nodeprob(tnode)-Pmis);
                 crit0 = impurity(tnode)*nodeprob(tnode);
                end
            else
                % crit0U = P(t0-tU)*mse(t0)
                % crit0  = P(t0)*mse(t0)
                %              Pmis = sum(Wnode(idxnan));
                Pmis = sum(Wsub(idxnan));
                crit0U = resuberr(tnode)*(nodeprob(tnode)-Pmis);
                crit0 = resuberr(tnode)*nodeprob(tnode);
            end

            % Find optimal split for this variable
            [critval,cutval] = classregtreeRCcritval(full(x),doclass,c,w,pratio,...
             xcat,Criterion,bestcrit,double(crit0U),minleaf);

            % Change best split if this one is best so far
            if critval>bestcrit
                bestcrit = critval;
                bestvar = jvar;
                bestcut = cutval;
            end
        end

        % Split this node using the best rule found
        % Note: we have leftside==~rightside in the absence of NaN's
        if bestvar~=0
            %nvarsplit2(bestvar) = nvarsplit2(bestvar)+1;
            %nvarsplit(bestvar) = nvarsplit(bestvar)+1;
            x = Xnode(:,bestvar);

            % Send observations left or right
            %if ~iscat2(bestvar)
            cutvar(tnode) = bestvar;
            leftside = x<bestcut;
            rightside = x>=bestcut;
            %else
            %   cutvar(tnode) = -bestvar;          % negative indicates cat. var. split
            %   leftside = ismember(x,bestcut{1});
            %   rightside = ismember(x,bestcut{2});
            %end

            % Store split position, children, parent, and node number
            cutpoint{tnode} = bestcut;
            children(tnode,:) = nextunusednode + (0:1);
            nodenumber(nextunusednode+(0:1)) = nextunusednode+(0:1)';
            parent(nextunusednode+(0:1)) = tnode;
            rpm(:,tnode) = promat(:,bestvar);
            if strcmp(mdiff,'all') || strcmp(mdiff,'node') && K > 1
                if ~isempty(md_idx)
                    isdelta(tnode) = bestvar <= max(md_idx);
                end
            end
         
            % Find surrogate splits
            if surrogate
                % tsurrvar is an array with indices of
                %   surrogate vars (association with best var above zero)
                %   found for this split, excluding the best variable itself.
                %   These indices are the original var indices in input data.
                %
                % tvarassoc and tvarimp are numeric arrays with var
                %   associations (must be positive) and var importance values
                %   for these surrogate splits.
                %
                % tsurrcut is a cell array with surrogate split cuts, same
                %   convention as for cutpoint.
                %
                % tsurrflip is a numeric array with 0's for categorical
                %   surrogate splits and either -1 or +1 for numeric surrogate
                %   splits. -1 for a numeric splits means that left and right
                %   must be swapped, that is, leftside=x>=cut and
                %   rightside=x<cut for this surrogate split.
                %
                % tvarassoc, tsurrcut, and tsurflip have length
                %   numel(tsurrvar).
                %
                % tvarimp has length numel(varmap). These are variable
                % importance contributions from this branch node for *all*
                % predictors, not only surrogate predictors with positive
                % measure of association.
                %
                % tleftORright is a 2D array of size Nt-by-numel(tsurrvar)
                %   with surrogate split indices for observations: -1 if the
                %   surrogate split sends an observation left, +1 if it sends
                %   an observation right, and 0 if uncertain.
                [tvarassoc,tvarimp,tsurrvar,tsurrcut,tsurrflip,tleftORright] = ...
                 findsurrogate(Xnode,Cnode,Wnode,Wtot,doclass,isimpurity,critfun,...
                 varmap,iscat,bestvar,Cost,resuberr(tnode),pratio,crit0,...
                 leftside,rightside);

                % Update variable importance for cuts on best variable
                varimp(varmap) = varimp(varmap) + tvarimp;

                % Sort vars by their associations with the best var.
                [~,idxvarsort] = sort(tvarassoc,'descend');

                % Store surrogate cuts and if they need to be flipped
                surrcut(tnode) = {tsurrcut(idxvarsort)};
                surrflip(tnode) = {tsurrflip(idxvarsort)};

                % Store variables for surrogate splits.
                % For categorical vars, store negative indices.
                tsurrvar = tsurrvar(idxvarsort);
                tiscat = iscat(tsurrvar);
                tsurrvar(tiscat) = -tsurrvar(tiscat);
                surrvar(tnode) = {tsurrvar};

                % Store variable associations
                varassoc(tnode) = {tvarassoc(idxvarsort)};

                % Append lists of observations to be assigned to left and
                % right children
                for jmis=1:length(idxvarsort)
                    idxmissing = (1:Nt)';
                    idxmissing = idxmissing(~(leftside | rightside));
                    if isempty(idxmissing)
                        break;
                    else
                        surrmissing = tleftORright(idxmissing,idxvarsort(jmis));
                        leftside(idxmissing(surrmissing<0)) = true;
                        rightside(idxmissing(surrmissing>0)) = true;
                    end
                end             
            end
         
            % Assign observations for the next node
            assignednode{nextunusednode} = noderows(leftside);
            assignednode{nextunusednode+1} = noderows(rightside);
         
            % Update next node index
            nextunusednode = nextunusednode+2;
        end
    end
    tnode = tnode + 1;
end

topnode        = nextunusednode - 1;
Tree.method    = Method;
Tree.node      = nodenumber(1:topnode);
Tree.parent    = parent(1:topnode);
Tree.class     = yfitnode(1:topnode);
Tree.var       = cutvar(1:topnode);
Tree.cut       = cutpoint(1:topnode);
Tree.children  = children(1:topnode,:);
Tree.nodeprob  = nodeprob(1:topnode);
Tree.nodeerr   = resuberr(1:topnode);
Tree.nodesize  = nodesize(1:topnode);
Tree.npred     = nvars;
Tree.catcols   = categ;
Tree.names     = names;
Tree.minleaf   = minleaf;
Tree.minparent = minparent;
%if strcmp(mdiff,'all') && K > 1
%    Tree.nvartosample = nvartosample + 1;
%else
    Tree.nvartosample = nvartosample;
%end
Tree.mergeleaves = Merge;
%Tree.nvarsplit = nvarsplit2;
%Tree.nvarsplit = nvarsplit;
Tree.rpm = rpm(:,1:topnode);  %Store proj matrices in a structure field
Tree.isdelta = isdelta(1:topnode);

if doclass
    Tree.prior     = Prior;
    Tree.nclasses  = nClasses;
    Tree.cost      = Cost;
    Tree.classprob = classprob(1:topnode,:);
    Tree.classcount= classcount(1:topnode,:);
    Tree.classname = cnames;
    if isimpurity
        Tree.impurity = impurity(1:topnode);
    end
    Tree.splitcriterion = Criterion;
else
    Tree.qetoler = qetoler;
end

% Get surrogate split info.
if ~isempty(surrvar)
    % Normalize var importance by the number of tree nodes.
    nbranch = sum(any(children(1:topnode,:)~=0,2));
    if nbranch>0
        Tree.varimp = varimp/nbranch;
    end

    % Throw away empty elements for surrogate splits.
    Tree.varassoc = varassoc(1:topnode);
    Tree.surrvar = surrvar(1:topnode);
    Tree.surrcut = surrcut(1:topnode);
    Tree.surrflip = surrflip(1:topnode);
end

if strcmpi(Merge,'on')
    Tree = mergeleaves(Tree); % merge leaves with same class
end

if strcmpi(Prune,'on')        % compute optimal pruning sequence if requested
    Tree = prune(Tree);
end
end

%----------------------------------------------------
function M = srpmat(d,k,method,varargin)
    if strcmp(method,'binary')
        rho = varargin{1};
        M = zeros(d,k);
        nnzs = round(k*d*rho);
        nzs=randperm(d*k,nnzs);
        npos = rand(nnzs,1) > 0.5;
        M(nzs(npos))=1;
        M(nzs(~npos))=-1;
        isnz = M~=0;
        OnlyNz = isnz & repmat(sum(isnz)==1,d,1);
        M(OnlyNz) = 1;
        M = sparse(unique(M(:,any(M))','rows','stable')');
    elseif strcmp(method,'binary-redundant')
        rho = varargin{1};
        M = zeros(d,k);
        nnzs = round(k*d*rho);
        nzs=randperm(d*k,nnzs);
        npos = rand(nnzs,1) > 0.5;
        M(nzs(npos))=1;
        M(nzs(~npos))=-1;
        M = sparse(M);
    elseif strcmp(method,'continuous')
        rho = varargin{1};
        M = zeros(d,k);
        nnzs = round(k*d*rho);
        nzs=randperm(d*k,nnzs);
        M(nzs) = rand(1,nnzs)*2 - 1;
        isnz = M~=0;
        OnlyNz = isnz & repmat(sum(isnz)==1,d,1);
        M(OnlyNz) = 1;
        M = sparse(unique(M(:,any(M))','rows','stable')');
    elseif strcmp(method,'binary-adjusted')
        rho = varargin{1};
        kk = varargin{3};
        M = zeros(d,kk);
        nnzs = round(kk*d*rho);
        nzs=randperm(d*kk,nnzs);
        npos = rand(nnzs,1) > 0.5;
        M(nzs(npos))=1;
        M(nzs(~npos))=-1;
        isnz = M~=0;
        OnlyNz = isnz & repmat(sum(isnz)==1,d,1);
        M(OnlyNz) = 1;
        M = unique(M(:,any(M))','rows','stable')';
        M = M(:,1:min(k,size(M,2)));
        M = sparse(M);
    elseif strcmp(method,'continuous-adjusted')
        rho = varargin{1};
        kk = varargin{3};
        M = zeros(d,kk);
        nnzs = round(kk*d*rho);
        nzs=randperm(d*kk,nnzs);
        M(nzs) = rand(1,nnzs)*2 - 1;
        isnz = M~=0;
        OnlyNz = isnz & repmat(sum(isnz)==1,d,1);
        M(OnlyNz) = 1;
        M = unique(M(:,any(M))','rows','stable')';
        M = M(:,1:min(k,size(M,2)));
        M = sparse(M);
    elseif strcmp(method,'frc')
        nmix = varargin{2};
        M = zeros(d,k);
%         p = 1;
%         for i = 1:nmix-1
%             p = p*(d-i)/d;
%         end
%         kk = round(4*k/p);
%         go = true;
%         while go
%             idx = randi(d,nmix,kk);
%             idx = idx(:,all(diff(sort(idx)),1));
%             go = size(idx,2) < k;
%         end
%         idx = idx(:,1:k);
        idx = randperms(d,nmix,k);
        idx = (ndgrid(1:k,1:nmix)'-1)*d + idx;
        M(idx) = rand(1,nmix*k)*2 - 1;
        M = sparse(M);
    elseif strcmp(method,'uniform-nnzs-binary')
        nmix = varargin{2};
        min_nmix = min(nmix);
        max_nmix = max(nmix);
        M = zeros(d,k);
        idx = randperms(d,max_nmix,k);
        idx = (ndgrid(1:k,1:max_nmix)'-1)*d + idx;
        nnzs = nmix(randi(length(nmix),1,k));
        for i = 1:length(nmix)
            idx(nmix(i)+1:end,nnzs==nmix(i)) = NaN;
        end
        idx(isnan(idx(:))) = [];
        ln = length(idx(:));
        ispos = rand(ln,1) > 0.5;
        M(idx(ispos)) = 1;
        M(idx(~ispos)) = -1;
        M = sparse(M);
    elseif strcmp(method,'uniform-nnzs-continuous')
        nmix = varargin{2};
        min_nmix = min(nmix);
        max_nmix = max(nmix);
        M = zeros(d,k);
        idx = randperms(d,max_nmix,k);
        idx = (ndgrid(1:k,1:max_nmix)'-1)*d + idx;
        nnzs = nmix(randi(length(nmix),1,k));
        for i = 1:length(nmix)
            idx(nmix(i)+1:end,nnzs==nmix(i)) = NaN;
        end
        idx(isnan(idx(:))) = [];
        M(idx(:)) = rand(1,length(idx(:)))*2 - 1;
        M = sparse(M);
    elseif strcmp(method,'poisson')
        lambda = varargin{2};
        M = zeros(d,k);
        go = true;
        while go
            nnzsPerCol = poissrnd(lambda,1,k);
            go = ~any(nnzsPerCol);
        end
        nnzsPerCol(nnzsPerCol > d) = d;
        nmix = unique(nnzsPerCol);
        max_nmix = nmix(end);
        idx = randperms(d,max_nmix,k);
        idx = repmat(0:k-1,max_nmix,1)*d + idx;
        for i = 1:length(nmix)
            idx(nmix(i)+1:end,nnzsPerCol==nmix(i)) = NaN;
        end
        one_nnz_idx = idx(:,nnzsPerCol==1);
        if isempty(one_nnz_idx)
            one_nnz_idx = [];
        else
            one_nnz_idx(isnan(one_nnz_idx)) = [];
        end
        idx(isnan(idx(:))) = [];
        nnzsTotal = length(idx(:));
        ispos = rand(nnzsTotal,1) > 0.5;
        M(idx(~ispos)) = -1;        
        M([idx(ispos),one_nnz_idx]) = 1;
%         M = sparse(M);
        M = sparse(unique(M(:,any(M))','rows','stable')');
    end
end

%----------------------------------------------------
function v=gdi(p)
%GDI Gini diversity index

v=1-sum(p.^2,2);
end

%----------------------------------------------------
function v=twoing(Pleft, P1, Pright, P2)
%TWOING Twoing index

v = 0.25 * Pleft .* Pright .* sum(abs(P1-P2),2).^2;
end

%----------------------------------------------------
function v=deviance(p)
%DEVIANCE Deviance

v = -2 * sum(p .* log(max(p,eps(class(p)))), 2);
end

% --------------------------------------
function Tree = mergeleaves(Tree)
%MERGELEAVES Merge leaves that originate from the same parent node and give
% the sum of risk values greater or equal to the risk associated with the
% parent node.

N = length(Tree.node);
isleaf = (Tree.var==0)';   % no split variable implies leaf node
isntpruned = true(1,N);
doprune = false(1,N);
Risk = risk(Tree)';
unsplitRisk = risk(Tree,1:N,'mode','unsplit');
adjfactor = (1 - 100*eps(class(Risk)));

% Work up from the bottom of the tree
while(true)
   % Find ''twigs'' with two leaf children
   branches = find(~isleaf & isntpruned);
   twig = branches(sum(isleaf(Tree.children(branches,:)),2) == 2);
   if isempty(twig)
      break;            % must have just the root node left
   end
   
   % Find twigs to ''unsplit'' if the error of the twig is no larger
   % than the sum of the errors of the children
   Rtwig = Risk(twig);
   kids = Tree.children(twig,:);
   Rsplit = unsplitRisk(twig) + sum(Risk(kids),2);
   unsplit = Rsplit >= Rtwig'*adjfactor;
   if any(unsplit)
      % Mark children as pruned, and mark twig as now a leaf
      isntpruned(kids(unsplit,:)) = 0;
      twig = twig(unsplit);   % only these to be marked on next 2 lines
      isleaf(twig) = 1;
      doprune(twig) = 1;
   else
      break;
   end
end

% Remove splits that are useless
if any(doprune)
   Tree = prune(Tree,'nodes',find(doprune));
end
end

% ------------------------------------
function idx = getclassindex(cnames,g)
%GETCLASSINDEX Find indices for class names in another list of names
%   IDX = GETCLASSINDEX(CNAMES,G) takes a list CNAMES of class names
%   (such as the grouping variable values in the treefit or classify
%   function) and another list G of group names (as might be supplied
%   in the ''prior'' argument to those functions), and finds the indices
%   of the CNAMES names in the G list.  CNAMES should be a cell array
%   of strings.  G can be numbers, a string array, or a cell array of
%   strings

% Convert to common string form, whether input is char, cell, or numeric
if isnumeric(g)
   g = cellstr(strjust(num2str(g(:)), 'left'));
elseif ~iscell(g)
   g = cellstr(g);
end

% Look up each class in the grouping variable.
[~,idx] = ismember(cnames,g);
end

% ---------------------------------------
function [varassoc,varimp,surrvar,surrcut,surrflip,leftORright] = ...
    findsurrogate(Xnode,Cnode,Wnode,Wtot,doclass,isimpurity,critfun,...
    varmap,iscat,bestvar,Cost,tresuberr,pratio,crit0,leftside,rightside)
% Get number of vars and make default output
nvar = length(varmap);
N = size(Xnode,1);
varassoc = zeros(1,nvar);
varimp = zeros(1,nvar);
surrcut = cell(1,nvar);
surrvar = false(1,nvar);
surrflip = zeros(1,nvar);
leftORright = zeros(N,nvar);

% Total weight in this node
Wt = sum(Wnode);

% Left and right probabilities for the best split
pL = sum(Wnode(leftside))/Wt;
pR = sum(Wnode(rightside))/Wt;
minp = min(pL,pR);

% Loop over variables
for ivar=1:nvar
    % Get the predictor from the original data X
    jvar = varmap(ivar);
 
    % If best-split variable, assign left and right cases.
    % Best variable is not a surrogate variable but we need to compute
    % varimp for it.
    if jvar==bestvar
        leftORright(leftside,ivar)  = -1;
        leftORright(rightside,ivar) = +1;
    else
        %
        % Find the split that maximizes pLL+pRR
        %
        x = Xnode(:,jvar);        
        
        % If categorical variable, add every category to the side with
        % larger probability
        if iscat(jvar)
            [grp,~,grpval] = grp2idx(x);
            Ngrp = max(grp);
            if Ngrp<2
                continue;
            end
            sendgrpleft = false(Ngrp,1);
            for igrp=1:Ngrp
                tf = grp==igrp;
                Wleft = sum(Wnode(tf & leftside));
                Wright = sum(Wnode(tf & rightside));
                if Wleft<Wright
                    leftORright(tf,ivar) = +1;
                else
                    leftORright(tf,ivar) = -1;
                    sendgrpleft(igrp) = true;
                end
            end
            leftvals = grpval(sendgrpleft);
            rightvals = grpval(~sendgrpleft);
            if isempty(leftvals) || isempty(rightvals)
                continue;
            end
            leftvals = leftvals(:)';
            rightvals = rightvals(:)';
            pLL = sum(Wnode(leftORright(:,ivar)<0 & leftside))/Wt;
            pRR = sum(Wnode(leftORright(:,ivar)>0 & rightside))/Wt;
            if minp>1-pLL-pRR && pLL>0 && pRR>0
                surrvar(ivar) = true;
                surrcut{ivar} = {leftvals rightvals};
                varassoc(ivar) = (minp-(1-pLL-pRR)) / minp;
            end
            
        % If numeric variable, try all splits
        else
            % Find NaN's and sort
            idxnotnan = find(~isnan(x));
            if isempty(idxnotnan)
                continue;
            end
            [x,idxsorted] = sort(x(idxnotnan));
            idx = idxnotnan(idxsorted);
            
            % Determine if there's anything to split along this variable
            maxeps = max(eps(x(1)), eps(x(end)));
            if x(1)+maxeps > x(end)
                continue;
            end
            
            % Accept only splits on rows with distinct values
            idxdistinct = find(x(1:end-1) + ...
                max([eps(x(1:end-1)) eps(x(2:end))],[],2) < x(2:end));
            if isempty(idxdistinct)
                continue;
            end
            idxdistinct(end+1) = length(x);
            
            % Group into left and right using optimal split
            w = repmat(Wnode(idx)/Wt,1,2);
            w(rightside(idx),1) = 0;
            w(leftside(idx),2) = 0;
            w(~rightside(idx) & ~leftside(idx), :) = 0;
            w = cumsum(w,1);
            w = w(idxdistinct,:);
            x = x(idxdistinct);
            
            % Find split maximizing pLL+pRR
            [wLLandRRmax,i1] = ...
                max(w(1:end-1,1)+w(end,2)-w(1:end-1,2));
            [wLRandRLmax,i2] = ...
                max(w(end,1)-w(1:end-1,1)+w(1:end-1,2));
            if wLLandRRmax<wLRandRLmax
                surrflip(ivar) = -1;
                pLL = w(end,1)-w(i2,1);
                pRR = w(i2,2);
                cut = 0.5*(x(i2)+x(i2+1));
            else
                surrflip(ivar) = +1;
                pLL = w(i1,1);
                pRR = w(end,2)-w(i1,2);
                cut = 0.5*(x(i1)+x(i1+1));
            end
            x = Xnode(:,jvar);
            leftORright(x<cut,ivar)  = -surrflip(ivar);
            leftORright(x>=cut,ivar) = +surrflip(ivar);
            
            % Get association
            if minp>1-pLL-pRR && pLL>0 && pRR>0
                surrvar(ivar) = true;
                surrcut{ivar} = cut;
                varassoc(ivar) = (minp-(1-pLL-pRR)) / minp;
            end
        end
    end
        
    % Compute var importance
    sendleft = leftORright(:,ivar)<0;
    sendright = leftORright(:,ivar)>0;
    Cleft = Cnode(sendleft,:);
    Cright = Cnode(sendright,:);
    Wleft = Wnode(sendleft);
    Wright = Wnode(sendright);
    idxmiss = ~(sendleft | sendright);
    if doclass
        Pleft = sum(bsxfun(@times,Cleft,Wleft),1);
        Pright = sum(bsxfun(@times,Cright,Wright),1);
        if isimpurity
            Pleft = Pleft.*pratio;
            Pright = Pright.*pratio;
            varimp(ivar) = (1-sum(Wnode(idxmiss))/Wt)*crit0 ...
                - sum(Pleft)*feval(critfun,Pleft/sum(Pleft)) ...
                - sum(Pright)*feval(critfun,Pright/sum(Pright));
        else
            varimp(ivar) = Wt*tresuberr ...
                - min(Pleft*Cost) - min(Pright*Cost);
        end
    else
        ybarleft = sum(Wleft.*Cleft)/sum(Wleft);
        ybarright = sum(Wright.*Cright)/sum(Wright);
        varimp(ivar) = (1-sum(Wnode(idxmiss))/Wt)*crit0 ...
            - ( sum(Wleft.*(Cleft-ybarleft).^2) ...
            + sum(Wright.*(Cright-ybarright).^2) )/Wtot;
    end
end

% Return only values for surrogate split vars (satisfying varassoc>0).
% varimp is the only exception - it keeps values for all variables.
varassoc = varassoc(surrvar);
surrcut = surrcut(surrvar);
surrflip = surrflip(surrvar);
leftORright = leftORright(:,surrvar);
surrvar = varmap(surrvar);
end


% ------------------------------------
function treeobj=struct2tree(treeobj,S)
% Copy fields from structure S to tree object treeobj

% Look at all fields required for regression or classification trees
allfields = {'method'   'node'     'parent'   'class'   'var' ...
             'cut'      'children' 'nodeprob' 'nodeerr' ...
             'nodesize' 'npred'    'catcols'  ...
             'nclasses' 'prior'    'cost'     ...
             'classprob' 'classcount' 'classname'};
fn = fieldnames(S);
if ~ismember('method',fn) || ...
   (strcmpi(S.method,'classification') && ~all(ismember(allfields,fn))) || ...
   (strcmpi(S.method,'regression')     && ~all(ismember(allfields(1:12),fn)))
   error(message('stats:classregtree:BadTree'));
end
if strcmpi(S.method,'regression')
    nrequired = 12;
else
    nrequired = numel(allfields);
end
for j=1:nrequired
    fname = allfields{j};
    treeobj.(fname) = S.(fname);
end

% Look at optional fields
optionalfields = {'names' 'prunelist' 'alpha' 'ntermnodes' ...
    'impurity' 'prunecriterion' ...
    'minparent' 'minleaf' 'nvartosample' 'mergeleaves' ...
    'splitcriterion' 'qetoler' 'varassoc' 'varimp' 'nvarsplit' ...
    'surrvar' 'surrcut' 'surrflip' 'catsplit'};
for j=1:numel(optionalfields)
    fname = optionalfields{j};
    if isfield(S,fname)
        treeobj.(fname) = S.(fname);
    end
end
end
