(function() {
  'use strict';

  angular
    .module('bubbleApp')
    .config(bubbleAppRoutes);

  bubbleAppRoutes.$inject = [
    '$stateProvider',
    '$urlRouterProvider'
  ];

  function bubbleAppRoutes($stateProvider, $urlRouterProvider) {
    $urlRouterProvider.otherwise('/');

    $stateProvider
      .state('home', {
      url: '/',
      controller: 'TweetListCtrl as vm',
      templateUrl: 'app/tweetlist/tweetlist.html'
    });
  }
})();